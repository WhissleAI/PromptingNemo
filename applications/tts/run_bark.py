from transformers import AutoProcessor, BarkModel
import scipy


processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark")

sample_rate = model.generation_config.sample_rate


voice_preset = "v2/en_speaker_6"

inputs = processor("The caller's hurried voice was barely audible over the street chaos blared, and passersby shuffled. Amidst the noise, the agent [clears throat] to listen attentively as the caller described the bouquet. Faint [music] played in the backdrop, adding a festive vibe. There was a [gasp] moment when the caller confirmed the delivery details, followed by a shared [laughter] over the anticipation of the mother's reaction. Finally, a [sigh] of relief marked the end of the call, with gratitude expressed. ", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
