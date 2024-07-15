import argparse
import ast
import csv
import os
from pathlib import Path

import numpy as np
import torch
from utils.tensorrtllm_utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   add_common_args, load_tokenizer, read_model_name,
                   throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--max_output_len', type=int, default=100)
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument('--output_log_probs_npy',
                        type=str,
                        help='Numpy file where the log_probs are stored',
                        default=None)
    parser.add_argument('--output_cum_log_probs_npy',
                        type=str,
                        help='Numpy file where the cum_log_probs are stored',
                        default=None)
    parser.add_argument(
        '--run_profiling',
        default=False,
        action='store_true',
        help="Run several 10 iterations to profile the inference latencies.")
    parser = add_common_args(parser)

    return parser.parse_args(args=args)


def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        for curr_text in input_text:
            if prompt_template is not None:
                curr_text = prompt_template.format(input_text=curr_text)
            input_ids = tokenizer.encode(curr_text,
                                         add_special_tokens=add_special_tokens,
                                         truncation=True,
                                         max_length=max_input_length)
            batch_input_ids.append(input_ids)
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_ids = np.array(line, dtype='int32')
                    batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8',
                      errors='replace') as txt_file:
                input_text = txt_file.readlines()
                batch_input_ids = tokenizer(
                    input_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)["input_ids"]
        else:
            print('Input file format not supported.')
            raise SystemExit

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids


def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 output_csv=None,
                 output_npy=None,
                 context_logits=None,
                 generation_logits=None,
                 cum_log_probs=None,
                 log_probs=None,
                 output_logits_npy=None,
                 output_cum_log_probs_npy=None,
                 output_log_probs_npy=None):
    batch_size, num_beams, _ = output_ids.size()
    if output_csv is None and output_npy is None:
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {batch_idx}]: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                print(
                    f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')
                return output_text

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
        np.save(output_file, outputs)

    # Save context logits
    if context_logits is not None and output_logits_npy is not None:
        context_logits = torch.cat(context_logits, axis=0)
        vocab_size_padded = context_logits.shape[-1]
        context_logits = context_logits.reshape([1, -1, vocab_size_padded])

        output_context_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_context"
        output_context_logits_file = Path(output_context_logits_npy)
        context_outputs = np.array(
            context_logits.squeeze(0).cpu().contiguous(),
            dtype='float32')  # [promptLengthSum, vocabSize]
        np.save(output_context_logits_file, context_outputs)

    # Save generation logits
    if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
        output_generation_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_generation"
        output_generation_logits_file = Path(output_generation_logits_npy)
        generation_outputs = np.array(generation_logits.cpu().contiguous(),
                                      dtype='float32')
        np.save(output_generation_logits_file, generation_outputs)

    # Save cum log probs
    if cum_log_probs is not None and output_cum_log_probs_npy is not None:
        cum_log_probs_file = Path(output_cum_log_probs_npy)
        cum_log_probs_outputs = np.array(cum_log_probs.cpu().contiguous(),
                                         dtype='float32')
        np.save(cum_log_probs_file, cum_log_probs_outputs)

    # Save cum log probs
    if log_probs is not None and output_log_probs_npy is not None:
        log_probs_file = Path(output_log_probs_npy)
        log_probs_outputs = np.array(log_probs.cpu().contiguous(),
                                     dtype='float32')
        np.save(log_probs_file, log_probs_outputs)


class TensorRT_LLM:
    def __init__(self, args, config):
        
        args.max_output_len = config['max_output_len']
        args.tokenizer_dir = config['tokenizer_dir']
        args.engine_dir = config['engine_dir']
        
        self.args = args
       
        
        self.batch_size = 1
        self.max_input_len = 100
        
        self.runner = self._initialize_model_runner()
        
        self.stop_words_list = None
        
        self.bad_words_list = None

    def _initialize_model_runner(self):
        self.runtime_rank = tensorrt_llm.mpi_rank()
        
        print("RUN TIME RANK", self.runtime_rank)
        
        logger.set_level(self.args.log_level)

        is_enc_dec = {
            name
            for name in os.listdir(self.args.engine_dir)
            if os.path.isdir(os.path.join(self.args.engine_dir, name))
        } == {'encoder', 'decoder'}

        self.model_name, self.model_version = read_model_name(self.args.engine_dir) if not is_enc_dec else ("", "")

        if self.args.tokenizer_dir is None:
            logger.warning(
                "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
            )
            self.args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[self.model_name]

        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=self.args.tokenizer_dir,
            vocab_file=self.args.vocab_file,
            model_name=self.model_name,
            model_version=self.model_version,
            tokenizer_type=self.args.tokenizer_type,
        )
        print("PAD ID", self.pad_id)

        runner_cls = ModelRunner if self.args.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=self.args.engine_dir,
            lora_dir=self.args.lora_dir,
            rank=self.runtime_rank,
            debug_mode=self.args.debug_mode,
            lora_ckpt_source=self.args.lora_ckpt_source,
            gpu_weights_percent=self.args.gpu_weights_percent,
        )

        if self.args.medusa_choices is not None:
            self.args.medusa_choices = ast.literal_eval(
                self.args.medusa_choices)
            assert self.args.temperature == 1.0, "Medusa should use temperature == 1.0"
            assert self.args.num_beams == 1, "Medusa should use num_beams == 1"
            runner_kwargs.update(medusa_choices=self.args.medusa_choices)

        if not self.args.use_py_session:
            runner_kwargs.update(
                max_batch_size=self.batch_size,
                max_input_len=self.max_input_len,
                max_output_len=self.args.max_output_len,
                max_beam_width=self.args.num_beams,
                max_attention_window_size=self.args.max_attention_window_size,
                sink_token_length=self.args.sink_token_length,
                max_tokens_in_paged_kv_cache=self.args.max_tokens_in_paged_kv_cache,
                kv_cache_enable_block_reuse=self.args.kv_cache_enable_block_reuse,
                kv_cache_free_gpu_memory_fraction=self.args.kv_cache_free_gpu_memory_fraction,
                enable_chunked_context=self.args.enable_chunked_context,
            )

        return runner_cls.from_dir(**runner_kwargs)
    
    def generate_output(self, input_text):
        
        batch_input_ids = parse_input(tokenizer=self.tokenizer,
                                  input_text=[input_text],
                                  prompt_template=None,
                                  add_special_tokens=self.args.add_special_tokens,
                                  max_input_length=self.args.max_input_length,
                                  pad_id=self.pad_id,
                                  num_prepend_vtokens=self.args.num_prepend_vtokens,
                                  model_name=self.model_name,
                                  model_version=self.model_version)
        
        input_lengths = [x.size(0) for x in batch_input_ids]
        encoder_input_lengths = None
        
        
        
        print(batch_input_ids)
        
        
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=batch_input_ids,
                encoder_input_ids=None,
                max_new_tokens=self.args.max_output_len,
                max_attention_window_size=self.args.max_attention_window_size,
                sink_token_length=self.args.sink_token_length,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                length_penalty=self.args.length_penalty,
                early_stopping=self.args.early_stopping,
                repetition_penalty=self.args.repetition_penalty,
                presence_penalty=self.args.presence_penalty,
                frequency_penalty=self.args.frequency_penalty,
                stop_words_list=self.stop_words_list,
                bad_words_list=self.bad_words_list,
                output_cum_log_probs=(self.args.output_cum_log_probs_npy != None),
                output_log_probs=(self.args.output_log_probs_npy != None),
                random_seed=self.args.random_seed,
                lora_uids=self.args.lora_task_uids,
                prompt_table=self.args.prompt_table_path,
                prompt_tasks=self.args.prompt_tasks,
                streaming=self.args.streaming,
                output_sequence_lengths=True,
                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                return_dict=True,
                medusa_choices=self.args.medusa_choices)
            torch.cuda.synchronize()
        
        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        context_logits = None
        generation_logits = None
        cum_log_probs = None
        log_probs = None
        if self.runner.gather_context_logits:
            context_logits = outputs['context_logits']
        if self.runner.gather_generation_logits:
            generation_logits = outputs['generation_logits']
        if self.args.output_cum_log_probs_npy != None:
            cum_log_probs = outputs['cum_log_probs']
        if self.args.output_log_probs_npy != None:
            log_probs = outputs['log_probs']
        output_text = print_output(self.tokenizer,
                        output_ids,
                        input_lengths,
                        sequence_lengths,
                        output_csv=self.args.output_csv,
                        output_npy=self.args.output_npy,
                        context_logits=context_logits,
                        generation_logits=generation_logits,
                        output_logits_npy=self.args.output_logits_npy,
                        cum_log_probs=cum_log_probs,
                        log_probs=log_probs,
                        output_cum_log_probs_npy=self.args.output_cum_log_probs_npy,
                        output_log_probs_npy=self.args.output_log_probs_npy)
        return output_text
        
        
if __name__ == '__main__':
    args = parse_arguments()
    model_processor = TensorRT_LLM(args)
    model_processor.generate_output("Where is mumbai?")
    #main(args)