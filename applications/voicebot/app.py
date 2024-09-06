from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware

from werkzeug.utils import secure_filename

import os
import re
import json
import yaml
from pydantic import BaseModel
from typing import List, Optional
from pydub import AudioSegment
from deepgram import Deepgram
import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor


from utils.asr_utils import *
#from utils.rag_utils import *
from utils.llm_utils import *
from utils.tts_utils import *
from utils.openai_utils import *
from utils.search_utils import *
from utils.tts_piper_utils import PiperSynthesizer, clean_text_for_piper

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

##checkout model shelf
def load_config():
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

config = load_config()

MODEL_SHELF_PATH = config['MODEL_SHELF_PATH']
DEEPGRAM_API_KEY = config['DEEPGRAM_API_KEY']
dg_client = Deepgram(DEEPGRAM_API_KEY)

ort_session_en_ner, model_tokenizer_en, filterbank_featurizer = create_ort_session(model_name="EN_ner_emotion_commonvoice", model_shelf=MODEL_SHELF_PATH)
ort_session_en_iot, model_tokenizer_en_iot, filterbank_featurizer = create_ort_session(model_name="speech-tagger_en_slurp-iot", model_shelf=MODEL_SHELF_PATH)
#ort_session_en_pos, model_tokenizer_en, filterbank_featurizer = create_ort_session(model_name="EN_pos_emotion_commonvoice", model_shelf=MODEL_SHELF_PATH)
ort_session_euro_ner, model_tokenizer_euro, filterbank_featurizer = create_ort_session(model_name="EURO_ner_emotion_commonvoice", model_shelf=MODEL_SHELF_PATH)
ort_session_euro_iot, model_tokenizer_euro_iot, filterbank_featurizer = create_ort_session(model_name="EURO_IOT_slurp", model_shelf=MODEL_SHELF_PATH)
#ort_session_en_noise, model_tokenizer_noise, filterbank_featurizer = create_ort_session(model_name="EN_noise_ner_commonvoice_50hrs", model_shelf=MODEL_SHELF_PATH)



##Visual LLM model

DEV_MODE = True

if DEV_MODE == False:
    from utils.tensorrtllm_multimodal_utils import MultiModalModelRunner
    from utils.tensorrtllm_utils import TensorRT_LLM

    multimodal_llm_hf_dir = config['MULTIMODAL_LLM']['hf_model_dir']
    multimodal_llm_engine_dir = config['MULTIMODAL_LLM']['llm_engine_dir']
    multimodal_llm_visual_engine_dir = config['MULTIMODAL_LLM']['visual_engine_dir']

    mutlimodal_runner = MultiModalModelRunner(multimodal_llm_hf_dir, multimodal_llm_engine_dir, multimodal_llm_visual_engine_dir)


    #llm_model_tensorrt = TensorRT_LLM(tllm_args,config['TENSORRT_LLM'])
    engine_dir = config['TENSORRT_LLM']['engine_dir']
    tokenizer_dir = config['TENSORRT_LLM']['tokenizer_dir']
    max_output_len = 100
    llm_model_tensorrt = TensorRT_LLM(engine_dir, tokenizer_dir, max_output_len)


    hf_api_token = config['HF_TOKEN']
    model_id = "google/gemma-2b-it"
    llm_model_hfapi = HuggingFaceAPI(model_id, hf_api_token)

    instructions = "Answer the following question accurately and concisely. Do not add additional queries or answers."
    conversation_history = [{"role": "system", "content": instructions}]

    xtts_model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_model = TextToSpeech(model_name=xtts_model_path)


tts_piper = PiperSynthesizer(MODEL_SHELF_PATH+"/piper/voices/en_US-amy-medium.onnx", 
                                    MODEL_SHELF_PATH+"/piper/configs/en_US-amy-medium.onnx.json", 
                                    length_scale=3)

async def transcribe_deepgram(file_path):
    async with dg_client.transcription.prerecorded({'buffer': file_path}, {'punctuate': True}) as response:
        return response

@app.post("/transcribe-web2")
async def transcribe_audio_onnx_web2(audio: UploadFile = File(...), model_name: str = Form('whissle'), Authorization: str = Header(...)):
    sessionid = Authorization.replace('Bearer ', '')
    message = await audio.read()
    cmd = ['ffmpeg', '-i', "-", '-ac', '1', '-ar','16000', '-f', 'wav', '-']
    cmd=" ".join(cmd)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    audio_file,error = await proc.communicate(message)
    
    transcript = ""
    token_timestamps = []

    if model_name == "deepgram":
        response = await transcribe_deepgram(BytesIO(audio_file))
        transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
        cleaned_string = transcript
        emotion_type = "neutral"
        entities_table = ""
    
    else:
        if model_name == "EN":
            loop = asyncio.get_event_loop()
            transcript, token_timestamps, duration = await loop.run_in_executor(executor, infer_audio_file, filterbank_featurizer, ort_session_en_ner, model_tokenizer_en, BytesIO(audio_file))
        elif model_name == "EURO":
            loop = asyncio.get_event_loop()
            transcript, token_timestamps, duration = await loop.run_in_executor(executor, infer_audio_file, filterbank_featurizer, ort_session_euro_ner, model_tokenizer_euro, BytesIO(audio_file))
        elif model_name == "EN-IOT":
            loop = asyncio.get_event_loop()
            transcript, token_timestamps, duration = await loop.run_in_executor(executor, infer_audio_file, filterbank_featurizer, ort_session_en_iot, model_tokenizer_en_iot, BytesIO(audio_file))
            transcript = transcript.replace("END", " END ")
            emotion = transcript.strip().split()[-1]

            return {
                'transcript': transcript,
                'token_timestamps': token_timestamps,
                'duration_seconds': duration,
                'emotion_type': emotion
            }
        elif model_name == "EURO-IOT":
            loop = asyncio.get_event_loop()
            transcript, token_timestamps = await loop.run_in_executor(executor, infer_audio_file, filterbank_featurizer, ort_session_euro_iot, model_tokenizer_euro_iot, audio_file)
            transcript = transcript.replace("END", " END ")
            emotion = transcript.strip().split()[-1]
        
            return {
                'transcript': transcript,
                'token_timestamps': token_timestamps,
                'duration_seconds': duration,
                'emotion_type': emotion
            }



        entities_table = extract_entities_web(transcript, token_timestamps, tag="NER")
        transcript = transcript.replace("END", " END ")
        cleaned_string, emotion_type = clean_string_and_extract_emotion(transcript)
    return {
        'transcript': transcript,
        'tagged_transcript': transcript,
        'emotion_type': emotion_type,
        'token_timestamps': token_timestamps,
        'entities_table': entities_table,
        'duration_seconds': duration
    }

# @app.post("/transcribe-s2s")
# async def transcribe_audio_onnx_s2s(audio: UploadFile = File(...), model_name: str = Form(...), language_id: str = Form(...)):
#     print("-----------------------New-Turn----------------------")
#     audio_file_name = secure_filename(audio.filename)
#     file_path = os.path.join('/workspace/advanced-speech-LLM-demo/voice-assistant/static/audio/input', audio_file_name)

#     save_directory = os.path.dirname(file_path)
#     os.makedirs(save_directory, exist_ok=True)
    
#     with open(file_path, 'wb') as temp_file:
#         temp_file.write(await audio.read())

#     if language_id == "EN":
#         if model_name == "ner_emotion_commonvoice":
#             transcript, token_timestamps = infer_audio_file(filterbank_featurizer, ort_session_en_ner, model_tokenizer_en, file_path)
#         elif model_name == "pos_emotion_commonvoice":
#             transcript, token_timestamps = infer_audio_file(filterbank_featurizer, ort_session_en_pos, model_tokenizer_en, file_path)
#         elif model_name == "ner_noise_commonvoice":
#             transcript, token_timestamps = infer_audio_file(filterbank_featurizer, ort_session_en_noise, model_tokenizer_noise, file_path)
#     elif language_id == "EURO":
#         transcript, token_timestamps = infer_audio_file(filterbank_featurizer, ort_session_euro_ner, model_tokenizer_euro, file_path)
    
#     transcript = transcript.replace("END", " END ")
#     transcript = re.sub(' +', ' ', transcript)
#     print("Generated tokens: ", transcript)
    
#     return {'transcript': transcript, 'token_timestamps': token_timestamps}

class LLMResponse(BaseModel):
    response: str
    input_text: str
    input_tokens: int
    output_tokens: int

class LLMRequest(BaseModel):
    content: str
    model_name: str
    emotion: str
    instruction: str

class LLMRequestWithSearch(BaseModel):
    content: str
    model_name: str
    emotion: str
    url: Optional[str] = ""
    searchengine: Optional[str] = ""
    system_instruction: Optional[str] = ""
    role: Optional[str] = ""
    conversation_history: Optional[List[dict]] = []
    # input_file is removed from here as we handle it separately



# Function without file upload
@app.post("/llm_response_without_file", response_model=LLMResponse)
async def llm_response_without_file(content: str = Form(...),
                                    model_name: str = Form(...),
                                    emotion: Optional[str] = Form(""),
                                    url: Optional[str] = Form(""),
                                    searchengine: Optional[str] = Form(""),
                                    system_instruction: Optional[str] = Form(""),
                                    role: Optional[str] = Form(""),
                                    conversation_history: Optional[str] = Form("")):

    print("URL:", url)
    print("Input Text:", content)
    print("Model Name:", model_name)
    
    print("Search Engine:", searchengine)
    print("Instruction:", system_instruction)
    print("History:", conversation_history)
    
    #role = "therapist"
    emotion = emotion.replace("EMOTION_", "")
    print("Emotion:", emotion)
    detailed_instruction = f"Considering your role as a {role} and understanding that the speaker is feeling {emotion}, provide an empathetic and context-aware response."
    
    print(detailed_instruction)
    print("Role:", role)

    # Parse conversation history
    if conversation_history:
        try:
            conversation_history = json.loads(conversation_history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid conversation history format")
    else:
        conversation_history = []

    if DEV_MODE:
        input_text = clean_tags(content) + f' {{emotionalstate: {emotion}}}'
        text, input_tokens, output_tokens = get_openai_response(input_text, detailed_instruction, conversation_history)
        return {"response": text, "input_text": content, "input_tokens": input_tokens, "output_tokens": output_tokens}
    else:
        if searchengine == 'duckduckgo':
            urls = search_duckduckgo(content, max_results=2)
            print("Fetching URLs:", urls)
            response = llm_model_tensorrt.generate_response_with_rag([content], instructions=detailed_instruction, history=conversation_history, urls=urls)
            text = response[0]
        else:
            if model_name == 'openai':
                input_text = clean_tags(content) + f' {{emotionalstate: {emotion}}}'
                text, input_tokens, output_tokens = get_openai_response(input_text, system_instruction, conversation_history)
                return {"response": text, "input_text": content, "input_tokens": input_tokens, "output_tokens": output_tokens}
            elif model_name == 'whissle':
                response = llm_model_tensorrt.generate_response([content], instructions=detailed_instruction, history=conversation_history, role=role)
                text = response[0]
                print("LLM text:", text)
            else:
                raise HTTPException(status_code=400, detail="Model not found")

            print("LLM text:", text)

    return {"response": text, "input_text": content}

@app.post("/llm_response_with_file", response_model=LLMResponse)
async def llm_response_with_file(content: str = Form(...),
                                 model_name: str = Form(...),
                                 emotion: Optional[str] = Form(""),
                                 url: Optional[str] = Form(""),
                                 searchengine: Optional[str] = Form(""),
                                 system_instruction: Optional[str] = Form(""),
                                 role: Optional[str] = Form(""),
                                 conversation_history: Optional[str] = Form(""),
                                 input_file: UploadFile = File(...)):

    print("URL:", url)
    print("Input Text:", content)
    print("Model Name:", model_name)
    print("Emotion:", emotion)
    print("Search Engine:", searchengine)
    print("Instruction:", system_instruction)
    print("History:", conversation_history)
    print("Input File:", input_file)
    
    detailed_instruction = f"Considering your role as a {role} and understanding that the speaker is feeling {emotion}, provide an empathetic and context-aware response."
    
    print(detailed_instruction)
    print("Role:", role)

    # Handle file upload
    save_directory = '/root/webaudio'
    os.makedirs(save_directory, exist_ok=True)
    filename = secure_filename(input_file.filename)
    file_path = os.path.join(save_directory, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(input_file.file, buffer)
    print(f"File saved at {file_path}")

    # Parse conversation history
    if conversation_history:
        try:
            conversation_history = json.loads(conversation_history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid conversation history format")
    else:
        conversation_history = []
    
    if DEV_MODE:
        text = f"This is a test response."
        


    
        
    else:

        # Generate caption for the input image
        test_image = mutlimodal_runner.load_test_image_local(file_path)
        result = mutlimodal_runner.run(test_image, "Question: What's the image about? Answer:")
        print("Image Caption:", result[0][0])
        caption = result[0][0]
        response = llm_model_tensorrt.generate_response([content], instructions=detailed_instruction, history=conversation_history, role=role, caption=caption)
        text = response[0]

    print("LLM text:", text)

    return {"response": text, "input_text": content}






@app.post("/llm_response_with_search", response_model=LLMResponse)
async def llm_response_with_search(content: str = Form(...),
                                   model_name: str = Form(...),
                                   emotion: str = Form(...),
                                   url: Optional[str] = Form(""),
                                   searchengine: Optional[str] = Form(""),
                                   system_instruction: Optional[str] = Form(""),
                                   role: Optional[str] = Form(""),
                                   conversation_history: Optional[str] = Form("")):  # Enforcing file upload

    print("URL:", url)
    print("Input Text:", content)
    print("Model Name:", model_name)
    print("Emotion:", emotion)
    print("Search Engine:", searchengine)
    print("Instruction:", system_instruction)
    print("History:", conversation_history)
    
    role = "therapist"
    print("Role:", role)



    # Parse conversation history
    if conversation_history:
        try:
            conversation_history = json.loads(conversation_history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid conversation history format")
    else:
        conversation_history = []

    if DEV_MODE:
        text = f"This is a test response."
    else:
        if url:
            test_image = mutlimodal_runner.load_test_image(args.hf_model_dir, url)
            result = mutlimodal_runner.run(test_image, "Question: Give a caption for this image. Answer:")
            print("Result:", result)
            text = result
        elif searchengine == 'duckduckgo':
            urls = search_duckduckgo(content, max_results=2)
            print("Fetching URLs:", urls)
            response = llm_model_tensorrt.generate_response_with_rag([content], instructions=system_instruction, history=conversation_history, urls=urls)
            text = response[0]
        else:
            if model_name == 'openai':
                input_text = clean_tags(content) + f' {{emotionalstate: {emotion}}}'
                text, input_tokens, output_tokens = get_openai_response(input_text, system_instruction, conversation_history)
            elif model_name == 'whissle':
                response = llm_model_tensorrt.generate_response([content], instructions=system_instruction, history=conversation_history, role=role)
                text = response[0]
            elif model_name == 'hf-gamma-2b-it':
                text = llm_model_hfapi._generate(content, max_length=50)
            else:
                raise HTTPException(status_code=400, detail="Model not found")

            print("LLM text:", text)

    return {"response": text, "input_text": content}


class RAGFileResponse(BaseModel):
    model_name: str
    url: str
    content: str
    emotion: Optional[str] = ""

@app.post("/rag_file_response")
async def rag_file_response(request: RAGFileResponse):
    try:
        model_name = request.model_name
        url = request.url
        query = request.content
        emotion = request.emotion

        print("Input Text: ", query)    
        print("model name", model_name)
        print("Emotion", emotion)
        print("RAG URL", url)
        
        rag_response = get_rag_response([url], query)
        return {"response": rag_response, "input_text": query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/generate_audio_cloning")
# async def generate_audio_cloning(text_to_convert: str = Form(...), output_filename: str = Form(...)):
#     try:
#         tts_model.tts_to_file_cloning(text_to_convert, speaker_wav_folder, language, output_filename)
#         return {"audio_file_path": output_filename}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_tts_audio")
async def upload_tts_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Process the audio file (e.g., convert to desired format, sample rate, etc.)
        audio = AudioSegment.from_file(file_location)
        processed_audio_location = f"processed/{file.filename}"
        audio.export(processed_audio_location, format="wav")

        return {"message": "File processed successfully", "file_location": processed_audio_location}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_audio_gtts")
async def generate_audio_gtts(text_to_convert: str = Form(...), output_filename: str = Form(...)):
    #try:
    tts = gTTS(text_to_convert)
    tts.save(output_filename)
    
    with open(output_filename, "rb") as audio_file:
        audio_content = audio_file.read()

    os.remove(output_filename)  # Clean up the saved file after reading its content

    return Response(content=audio_content, media_type="audio/mpeg")

    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))

    
    
@app.post("/generate_audio")
async def generate_audio_xtts(model_name: str = Form("piper"), text_to_convert: str = Form(...), output_filename: str = Form(...), ref_file: UploadFile = File(None)):
    
    print("TTS Model Name", model_name)
    
    if model_name == "xtts":
        if ref_file:
            audio_file_name = secure_filename(ref_file.filename)
            
            save_directory = '/root/webaudio'
            os.makedirs(save_directory, exist_ok=True)
            temp_file_path = os.path.join(save_directory, audio_file_name)
            
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(await ref_file.read())
        
            ref_file_path = os.path.join(save_directory, os.path.splitext(audio_file_name)[0] + '.wav')
        else:
            ref_file_path = None
        xtts_model.infer(text= text_to_convert, language='en', file_path=output_filename, speaker_wav_file_path=ref_file_path)
        
        with open(output_filename, "rb") as audio_file:
            audio_content = audio_file.read()
        duration_seconds = AudioSegment(audio_content).duration_seconds
        os.remove(output_filename)  # Clean up the saved file after reading its content
        return Response(content=audio_content, media_type="audio/mpeg", headers={"X-Duration": str(duration_seconds)})
    elif model_name == "piper":
        
        text_to_convert = clean_text_for_piper(text_to_convert)
        audio_content = tts_piper.synthesize(text_to_convert)
        duration_seconds = AudioSegment(audio_content).duration_seconds
        return Response(content=audio_content, media_type="audio/mpeg", headers={"X-Duration": str(duration_seconds)})


    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))

class ProcessTranscriptRequest(BaseModel):
    text: str
    model_name: str
    timestamps: str

@app.post("/process_transcript")
async def process_transcript(request: ProcessTranscriptRequest):
    input_text = request.text
    selected_model = request.model_name
    token_timestamps = request.timestamps
    
    if selected_model == "ner_emotion_commonvoice":
        processed_output = extract_entities(input_text, token_timestamps, tag="NER")
    else:
        processed_output = extract_entities(input_text, token_timestamps, tag="POS")
    
    print("Processed Output", processed_output)
    return {"processed_output": processed_output}

def clean_tags(input_text):
    input_text = input_text.split()
    new_sent = [word for word in input_text if "NER_" not in word and "END" not in word and "EMOTION_" not in word]
    return " ".join(new_sent)

if __name__ == '__main__':
    import uvicorn
    from uvicorn import Config, Server

    config = Config(app, host='127.0.0.1', port=5000, workers=10)
    server = Server(config)
    server.run()
