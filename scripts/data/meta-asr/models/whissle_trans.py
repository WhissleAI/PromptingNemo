from whissle import WhissleClient
import asyncio


async def transcribe_with_whissle(audio_path, client, ):
   """
   Transcribe audio using Whissle.


   Args:
       audio_path (str): Path to the audio file.
       client (WhissleClient): An instance of the WhissleClient.
       model_name (str): The model name to use for transcription.


   Returns:
       str: The transcribed text.
   """
   response = await client.speech_to_text(
       audio_file_path=audio_path,
       model_name=model_name
   )
   transcript = response.transcript
   return transcript
client = WhissleClient(auth_token='5fa8b1dfb042419e')
transcription = asyncio.run(transcribe_with_whissle("/hydra2-prev/home/compute/workspace_himanshu/whissle_python_api/examples/data/sample.wav", client))
print(transcription)