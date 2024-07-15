
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests

import torch
import tensorrt as trt

from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                          Blip2Processor, NougatProcessor, NougatTokenizerFast)

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo

# sys.path.append(str(Path(__file__).parent.parent))
# from enc_dec.run import TRTLLMEncDecModel

def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


class MultiModalModel:

    def __init__(self, hf_model_dir, llm_engine_dir, visual_engine_dir, args):
        self.hf_model_dir = hf_model_dir
        self.llm_engine_dir = llm_engine_dir
        self.visual_engine_dir = visual_engine_dir
        self.args = args
        self.device = self.set_device()
        self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)
        self.init_image_encoder()
        self.init_tokenizer()
        self.init_llm()

    def set_device(self):
        runtime_rank = tensorrt_llm.mpi_rank()
        device_id = runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        return f"cuda:{device_id}"

    def init_tokenizer(self):
        if self.args.nougat:
            self.tokenizer = NougatTokenizerFast.from_pretrained(self.hf_model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_dir, use_fast=False, use_legacy=False)

        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_image_encoder(self):
        vision_encoder_path = os.path.join(self.visual_engine_dir, 'visual_encoder_fp16.engine')
        logger.info(f'Loading engine from {vision_encoder_path}')
        with open(vision_encoder_path, 'rb') as f:
            engine_buffer = f.read()
        logger.info(f'Creating session from engine {vision_encoder_path}')
        self.visual_encoder_session = Session.from_serialized_engine(engine_buffer)

    def init_llm(self):
        if self.args.decoder_llm:
            self.model = ModelRunner.from_dir(self.llm_engine_dir,
                                              rank=tensorrt_llm.mpi_rank(),
                                              debug_mode=False,
                                              stream=self.stream)
            self.model_config = self.model.session._model_config
            self.runtime_mapping = self.model.session.mapping
        else:
            self.model = TRTLLMEncDecModel.from_engine(
                os.path.basename(self.hf_model_dir),
                self.llm_engine_dir,
                skip_encoder=self.args.nougat,
                debug_mode=False,
                stream=self.stream)

            if self.args.nougat:
                self.model_config = self.model.decoder_model_config
                self.runtime_mapping = self.model.decoder_runtime_mapping
            else:
                self.model_config = self.model.encoder_model_config
                self.runtime_mapping = self.model.encoder_runtime_mapping

    def generate(self, pre_prompt, post_prompt, image, decoder_input_ids, max_new_tokens, warmup):
        if not warmup:
            profiler.start("Generate")
            profiler.start("Vision")
        visual_features, visual_atts = self.get_visual_features(image)
        if not warmup:
            profiler.stop("Vision")

        pre_input_ids = self.tokenizer(pre_prompt, return_tensors="pt", padding=True).input_ids
        if post_prompt[0] is not None:
            post_input_ids = self.tokenizer(post_prompt, return_tensors="pt", padding=True).input_ids
            length = pre_input_ids.shape[1] + post_input_ids.shape[1] + visual_atts.shape[1]
        else:
            post_input_ids = None
            length = pre_input_ids.shape[1] + visual_atts.shape[1]

        input_lengths = torch.IntTensor([length] * self.args.batch_size).to(torch.int32)
        input_ids, ptuning_args = self.setup_fake_prompts(visual_features, pre_input_ids, post_input_ids, input_lengths)

        if warmup and self.args.decoder_llm and tensorrt_llm.mpi_rank() == 0:
            prompt_table = ptuning_args[0]
            prompt_table = torch.stack([prompt_table])
            np.save('prompt_table.npy', torch_to_numpy(prompt_table))
        if warmup:
            return None

        profiler.start("LLM")
        if self.args.decoder_llm:
            end_id = self.tokenizer.eos_token_id
            if 'opt' in self.hf_model_dir and self.args.blip2_encoder:
                end_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]

            output_ids = self.model.generate(
                input_ids,
                sampling_config=None,
                prompt_table_path='prompt_table.npy',
                max_new_tokens=max_new_tokens,
                end_id=end_id,
                pad_id=self.tokenizer.pad_token_id,
                top_k=self.args.top_k,
                num_beams=self.args.num_beams,
                output_sequence_lengths=False,
                return_dict=False)
        else:
            if self.args.nougat:
                ids_shape = (self.args.batch_size, visual_features.shape[1])
                input_ids = torch.zeros(ids_shape, dtype=torch.int32)

            output_ids = self.model.generate(
                input_ids,
                decoder_input_ids,
                max_new_tokens,
                num_beams=self.args.num_beams,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                debug_mode=False,
                prompt_embedding_table=ptuning_args[0],
                prompt_tasks=ptuning_args[1],
                prompt_vocab_size=ptuning_args[2])

            input_lengths = torch.ones(input_lengths.shape, dtype=input_lengths.dtype)
        profiler.stop("LLM")

        if tensorrt_llm.mpi_rank() == 0:
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True)
                for batch_idx in range(self.args.batch_size)
            ]

            stripped_text = [[output_beams_list[batch_idx][beam_idx].strip() for beam_idx in range(self.args.num_beams)]
                             for batch_idx in range(self.args.batch_size)]
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None

    def get_visual_features(self, image):
        visual_features = {'input': image.half()}
        visual_output_info = self.visual_encoder_session.infer_shapes([TensorInfo('input', trt.DataType.HALF, image.shape)])
        visual_outputs = {
            t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device=image.device)
            for t in visual_output_info
        }

        ok = self.visual_encoder_session.run(visual_features, visual_outputs, self.stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self.stream.synchronize()

        image_embeds = visual_outputs['output']
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        return image_embeds, image_atts

    def setup_fake_prompts(self, visual_features, pre_input_ids, post_input_ids, input_lengths):
        fake_prompt_id = torch.arange(self.model_config.vocab_size,
                                      self.model_config.vocab_size + visual_features.shape[0] * visual_features.shape[1])
        fake_prompt_id = fake_prompt_id.reshape(visual_features.shape[0], visual_features.shape[1])

        if post_input_ids is not None:
            input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
        else:
            input_ids = [fake_prompt_id, pre_input_ids]
        input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)

        if self.args.decoder_llm or self.runtime_mapping.is_first_pp_rank():
            ptuning_args = self.ptuning_setup(visual_features, input_ids, input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        hidden_size = self.model_config.hidden_size * self.runtime_mapping.tp_size
        if prompt_table is not None:
            task_vocab_size = torch.tensor([prompt_table.shape[1]], dtype=torch.int32).cuda()
            prompt_table = prompt_table.view((prompt_table.shape[0] * prompt_table.shape[1], prompt_table.shape[2]))

            assert prompt_table.shape[1] == hidden_size, "Prompt table dimensions do not match hidden size"

            prompt_table = prompt_table.cuda().to(dtype=tensorrt_llm._utils.str_dtype_to_torch(self.model_config.dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if self.model_config.remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)], dtype=torch.int32).cuda()
            if self.args.decoder_llm:
                tasks = tasks.unsqueeze(0)
        else:
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]


class MultiModalModelRunner:

    def __init__(self, hf_model_dir=None, llm_engine_dir=None, visual_engine_dir=None, max_new_tokens=30, batch_size=1,
                 log_level='info', decoder_llm=True, blip2_encoder=True, nougat=False, input_text="Question: which city is this? Answer:",
                 num_beams=1, top_k=1, run_profiling=False, check_accuracy=False):
        self.hf_model_dir = hf_model_dir
        self.llm_engine_dir = llm_engine_dir
        self.visual_engine_dir = visual_engine_dir
        self.args = argparse.Namespace(
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            log_level=log_level,
            visual_engine_dir=visual_engine_dir,
            llm_engine_dir=llm_engine_dir,
            hf_model_dir=hf_model_dir,
            decoder_llm=decoder_llm,
            blip2_encoder=blip2_encoder,
            nougat=nougat,
            input_text=input_text,
            num_beams=num_beams,
            top_k=top_k,
            run_profiling=run_profiling,
            check_accuracy=check_accuracy
        )
        self.model = MultiModalModel(hf_model_dir, llm_engine_dir, visual_engine_dir, self.args)

    def load_test_image_url(self, img_url):
        
        #img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
        image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

        return image
    
    def load_test_image_local(self, img_path):
        
        #img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
        image = Image.open(img_path).convert('RGB')

        return image
    
    

    def prepare_inputs(self, image, input_text):
        if self.args.blip2_encoder:
            model_type = 'Salesforce/blip2-opt-2.7b' if 'opt-2.7b' in self.hf_model_dir else 'Salesforce/blip2-flan-t5-xl'
            processor = Blip2Processor.from_pretrained(model_type)
            image = processor(image, input_text, return_tensors="pt")['pixel_values']
            pre_prompt, post_prompt = input_text, None

        elif self.args.nougat:
            processor = NougatProcessor.from_pretrained(self.hf_model_dir)
            image = processor(image, return_tensors="pt")['pixel_values']
            pre_prompt, post_prompt = input_text, None

        else:
            pre_prompt = "USER:\n" if "llava" in self.hf_model_dir else "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
            post_prompt = input_text + " ASSISTANT:"
            processor = AutoProcessor.from_pretrained(self.hf_model_dir)
            image = processor(text=input_text, images=image, return_tensors="pt")['pixel_values']

        return [pre_prompt] * self.args.batch_size, [post_prompt] * self.args.batch_size, image.expand(self.args.batch_size, -1, -1, -1).contiguous()

    def run(self, image, input_text):
        pre_prompt, post_prompt, image = self.prepare_inputs(image, input_text)
        image = image.to(self.model.device)

        if self.args.decoder_llm:
            decoder_input_ids = None
        else:
            config = AutoConfig.from_pretrained(self.hf_model_dir)
            decoder_start_id = config.decoder_start_token_id or config.decoder.bos_token_id
            decoder_input_ids = torch.IntTensor([[decoder_start_id]]).repeat((self.args.batch_size, 1))

        self.model.generate(pre_prompt, post_prompt, image, decoder_input_ids, self.args.max_new_tokens, warmup=True)
        tensorrt_llm.mpi_barrier()

        num_iters = 20 if self.args.run_profiling else 1
        for _ in range(num_iters):
            stripped_text = self.model.generate(pre_prompt, post_prompt, image, decoder_input_ids, self.args.max_new_tokens, warmup=False)

        runtime_rank = tensorrt_llm.mpi_rank()
        if runtime_rank == 0:
            return stripped_text


def multimodal_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--visual_engine_dir', type=str, required=True, help='Directory containing visual TRT engines')
    parser.add_argument('--llm_engine_dir', type=str, required=True, help='Directory containing TRT-LLM engines')
    parser.add_argument('--hf_model_dir', type=str, required=True, help="Directory containing tokenizer")
    parser.add_argument('--decoder_llm', action='store_true', help='Whether LLM is decoder-only or an encoder-decoder variant?')
    parser.add_argument('--blip2_encoder', action='store_true', help='Whether visual encoder is a BLIP2 model')
    parser.add_argument('--nougat', action='store_true', help='Run nougat pipeline')
    parser.add_argument('--input_text', type=str, default="Question: which city is this? Answer:", help='Text prompt to LLM')
    parser.add_argument('--num_beams', type=int, help="Use beam search if num_beams >1", default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--run_profiling', action='store_true', help='Profile runtime over several iterations')
    parser.add_argument('--check_accuracy', action='store_true', help='Check correctness of text output')

    return parser.parse_args()


if __name__ == '__main__':
    #args = multimodal_arguments()
    runner = MultiModalModelRunner(
        hf_model_dir="/external2/models/hf/opt-2.7b",
        llm_engine_dir="/external2/models/engine/opt-2.7b",
        visual_engine_dir="/external2/models/visual_engine/opt-2.7b/opt-2.7b"
    )

    # Example usage:
    img_url = "https://api.parashospitals.com/uploads/2019/02/food-good-for-liver-health.jpg"
    test_image = runner.load_test_image(img_url)
    
    input_text = "Question: Give a discriptive caption for this image. Answer:"

    result = runner.run(test_image, input_text)
    if result:
        print(result)
