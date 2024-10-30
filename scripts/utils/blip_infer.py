import onnxruntime, requests
from PIL import Image
import numpy as np
import torch
from transformers import BlipProcessor

vision_model_sess = onnxruntime.InferenceSession('/disk1/artifacts/whissle/model_shelf/blip/vision_model.onnx')
text_model_sess = onnxruntime.InferenceSession('/disk1/artifacts/whissle/model_shelf/blip/text_decoder_model.onnx')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

img_url = 'https://i.ytimg.com/vi/7C2914WckJ0/maxresdefault.jpg'

raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

new_size = (384, 384)
raw_image = raw_image.resize(new_size)

norm_d =  {

  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],

  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ]
}

image_array = np.array(raw_image) / 255.0
mean = norm_d["image_mean"]
std = norm_d["image_std"]
normalized_image = (image_array - mean) / std
original_image = Image.fromarray((normalized_image * 255).astype(np.uint8))

normalized_image = np.transpose(normalized_image, (2, 0, 1)).reshape(1, 3, 384, 384)

input_data = normalized_image.astype(np.float32)  # Your input data
input_name = vision_model_sess.get_inputs()[0].name
inputs_val = {input_name: input_data}

output_name = vision_model_sess.get_outputs()[0].name
result = vision_model_sess.run([output_name], inputs_val)
output_data = result[0]

image_embeds =torch.from_numpy(output_data.astype(np.float32))
image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

input_ids = np.array([[0]])
input_id_attention =  np.ones(input_ids.shape).astype(np.int64)

for i in range(20):
    inputs_val = {text_model_sess.get_inputs()[0].name:input_ids, text_model_sess.get_inputs()[1].name: input_id_attention , text_model_sess.get_inputs()[2].name: np.array(image_embeds), text_model_sess.get_inputs()[3].name:np.array(image_attention_mask)}
    result = text_model_sess.run(None, inputs_val)
    output_logits = result[0]

    # Example: Convert logits to probabilities or decode output
    probabilities = np.exp(output_logits) / np.sum(np.exp(output_logits), axis=-1, keepdims=True)
    pred_class = np.argmax(probabilities, axis=-1)
    input_ids = np.append(input_ids, [[pred_class[0][-1]]], axis=-1)
    input_id_attention =  np.ones(input_ids.shape).astype(np.int64)

print(processor.decode(input_ids[0], skip_special_tokens=True))