import numpy as np
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

VISION_MODEL_ONNX = 'vision_model.onnx'
vision_model = model.vision_model
vision_model.eval()

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
text = None

inputs = processor(raw_image, text, return_tensors="pt")#.to("cuda")

import torch
with torch.no_grad():
    vision_outputs = vision_model(inputs["pixel_values"])

image_embeds = vision_outputs[0]

with torch.no_grad():
  torch.onnx.export(vision_model, inputs["pixel_values"], VISION_MODEL_ONNX, input_names=["pixel_values"])

TEXT_DECODER_ONNX = 'text_decoder_model.onnx'
text_decoder_model = model.text_decoder
text_decoder_model.eval()

input_ids = torch.from_numpy(np.array([[0]]))
input_id_attention =  torch.ones(input_ids.size(), dtype=torch.long)

image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
input_dict = {"input_ids": input_ids, "attention_mask": input_id_attention, "encoder_hidden_states": image_embeds, "encoder_attention_mask": image_attention_mask}
# specify variable length axes
dynamic_axes = {"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}}
# export PyTorch model to ONNX
with torch.no_grad():
    torch.onnx.export(text_decoder_model, input_dict, TEXT_DECODER_ONNX, input_names=list(input_dict), dynamic_axes=dynamic_axes)

