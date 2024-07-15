import torch
import numpy as np
import onnxruntime
from nemo.collections.nlp.models import MTEncDecModel

# Load the model form pretrained:
model = MTEncDecModel.from_pretrained('nmt_es_en_transformer12x2')

# Export all model components to onnx:
model.encoder.export('encoder.onnx')
model.decoder.export('decoder.onnx')
model.log_softmax.export('classifier.onnx')

# Initialise all the onnx sessions:
encoder_session = onnxruntime.InferenceSession('encoder.onnx', providers=['CUDAExecutionProvider'])
decoder_session = onnxruntime.InferenceSession('decoder.onnx', providers=['CUDAExecutionProvider'])
classifier_session = onnxruntime.InferenceSession('classifier.onnx', providers=['CUDAExecutionProvider'])

# Preprocess the data using the original nemo model for simplicity:
TEXT = ['They are not even 100 metres apart: On Tuesday, the new B 33 pedestrian lights in Dorfparkplatz in Gutach became operational - within view of the existing Town Hall traffic lights.']
src_ids, src_mask = model.prepare_inference_batch(TEXT)
src_ids = src_ids.cpu().numpy()  # Convert to numpy for use with onnx
src_mask = src_mask.cpu().numpy().astype(int)

# Compute encoder hidden state:
encoder_input = {'input_ids': src_ids, 'encoder_mask': src_mask}
encoder_hidden_state = encoder_session.run(['last_hidden_states'], encoder_input)[0]

# Simple greedy search:
MAX_GENERATION_DELTA = 5
BOS = model.encoder_tokenizer.bos_id
EOS = model.encoder_tokenizer.eos_id
PAD = model.encoder_tokenizer.pad_id

def decode(tgt: np.array, embeding: np.array, src_mask: np.array) -> np.array:
      decoder_input = {
          'input_ids': tgt,
          'decoder_mask': (tgt != PAD).astype(np.int64),
          'encoder_mask': embeding,
          'encoder_embeddings': src_mask
      }
      decoder_hidden_state = decoder_session.run(['last_hidden_states'], decoder_input)[0]
      log_probs = classifier_session.run(['log_probs'], {'hidden_states': decoder_hidden_state})[0]
      return log_probs

max_out_len = encoder_hidden_state.shape[1] + MAX_GENERATION_DELTA
tgt=np.full(shape=encoder_hidden_state.shape[:-1], fill_value=0)
tgt[:, 0] = BOS

for i in range(1, max_out_len):
    log_probs = decode(tgt[:, :i], encoder_hidden_state, src_mask)
    next_tokens = log_probs[:, -1].argmax(axis=1) # NOTE: ONNX decoder returns multiple outputs which is different to pytorch version, so I get the last one (this could be where error is?)
    tgt[:, i] = next_tokens
    if ((tgt == EOS).sum(axis=1) > 0).all():
        break

tgt_torch = torch.from_numpy(tgt).to('cuda:0')
onnx_translation = model.ids_to_postprocessed_text(tgt_torch, model.decoder_tokenizer, model.target_processor)