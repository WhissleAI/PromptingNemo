from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from gunicorn.app.base import BaseApplication


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
import riva.client
from google.protobuf.json_format import MessageToDict
import requests
from dotenv import load_dotenv

from utils.asr_utils import *
from utils.rag_utils import *
from utils.qdrant_rag_utils import *
# from utils.llm_utils import *
from utils.blip_utils import *
from utils.tts_utils import *
from utils.openai_utils import *
from utils.mt_utils import *
from utils.search_utils import *
from utils.riva_utils import get_transcript, transform_riva_output
from utils.tts_piper_utils import PiperSynthesizer, clean_text_for_piper
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

app = FastAPI(redoc_url=None)

executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load configuration from environment variables
MODEL_SHELF_PATH = os.getenv('MODEL_SHELF_PATH')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
dg_client = Deepgram(DEEPGRAM_API_KEY)

# news_llm = HFLanguageModel(model_name_or_path='RedHenLabs/news-reporter-euro-3b')

# ort_session_en_ner, model_tokenizer_en, filterbank_featurizer = create_ort_session(model_name="EN_noise_ner_commonvoice_50hrs", model_shelf=MODEL_SHELF_PATH)
# ort_session_en_iot, model_tokenizer_en_iot, filterbank_featurizer = create_ort_session(model_name="speech-tagger_en_slurp-iot", model_shelf=MODEL_SHELF_PATH)
#ort_session_en_pos, model_tokenizer_en, filterbank_featurizer = create_ort_session(model_name="EN_pos_emotion_commonvoice", model_shelf=MODEL_SHELF_PATH)
# ort_session_euro_ner, model_tokenizer_euro, filterbank_featurizer = create_ort_session(model_name="EURO_ner_emotion_commonvoice", model_shelf=MODEL_SHELF_PATH)
# ort_session_euro_iot, model_tokenizer_euro_iot, filterbank_featurizer = create_ort_session(model_name="EURO_IOT_slurp", model_shelf=MODEL_SHELF_PATH)
#ort_session_en_noise, model_tokenizer_noise, filterbank_featurizer = create_ort_session(model_name="EN_noise_ner_commonvoice_50hrs", model_shelf=MODEL_SHELF_PATH)

#ort_session_ambernet, filterbank_featurizer, labels = load_ambernet_model_config('/projects/svanga/PromptingNemo/scripts/utils/ambernet_onnx/model.onnx', '/projects/svanga/asr_models/nemo/langid_ambernet_v1.12.0/model_config.yaml')

vision_model_sess, text_model_sess, blip_processor = create_blip_ort_session(model_name="blip",model_shelf=MODEL_SHELF_PATH)
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

asr_models = json.loads(os.getenv('ASR_MODELS'))

lang_to_model_map = {
    'en' : 'en-US-0.6b',
    'ru' : 'ru-RU-110m',
    'zh' : 'zh-CN'
}

##Visual LLM model

DEV_MODE = True

if DEV_MODE == False:
    from utils.tensorrtllm_multimodal_utils import MultiModalModelRunner
    from utils.tensorrtllm_utils import TensorRT_LLM

    multimodal_llm_hf_dir = os.getenv('MULTIMODAL_LLM_HF_MODEL_DIR')
    multimodal_llm_engine_dir = os.getenv('MULTIMODAL_LLM_LLM_ENGINE_DIR')
    multimodal_llm_visual_engine_dir = os.getenv('MULTIMODAL_LLM_VISUAL_ENGINE_DIR')

    mutlimodal_runner = MultiModalModelRunner(multimodal_llm_hf_dir, multimodal_llm_engine_dir, multimodal_llm_visual_engine_dir)


    #llm_model_tensorrt = TensorRT_LLM(tllm_args,config['TENSORRT_LLM'])
    engine_dir = os.getenv('TENSORRT_LLM_ENGINE_DIR')
    tokenizer_dir = os.getenv('TENSORRT_LLM_TOKENIZER_DIR')
    max_output_len = 100
    llm_model_tensorrt = TensorRT_LLM(engine_dir, tokenizer_dir, max_output_len)

    hf_api_token = os.getenv('HF_TOKEN')
    model_id = "google/gemma-2b-it"
    llm_model_hfapi = HuggingFaceAPI(model_id, hf_api_token)

    instructions = "Answer the following question accurately and concisely. Do not add additional queries or answers."
    conversation_history = [{"role": "system", "content": instructions}]

xtts_model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
xtts_model = TextToSpeech(model_name=xtts_model_path)

piper_models_config = {
    "en-US": {
        "model_path": "/piper/voices/en_US-amy-medium.onnx",
        "json_path": "/piper/configs/en_US-amy-medium.onnx.json"
    },
    "ru": {
        "model_path": "/piper/voices/ru_RU-dmitri-medium.onnx",
        "json_path": "/piper/configs/ru_RU-dmitri-medium.onnx.json"
    },
    "es": {
        "model_path": "/piper/voices/es_ES-davefx-medium.onnx",
        "json_path": "/piper/configs/es_ES-davefx-medium.onnx.json"
    },
    "fr": {
        "model_path": "/piper/voices/fr_FR-mls-medium.onnx",
        "json_path": "/piper/configs/fr_FR-mls-medium.onnx.json"
    },

}

piper_models = {}

for key in piper_models_config:
    print(MODEL_SHELF_PATH+piper_models_config[key]['json_path'])
    piper_models[key] = PiperSynthesizer(MODEL_SHELF_PATH+piper_models_config[key]['model_path'], 
                                    MODEL_SHELF_PATH+piper_models_config[key]['json_path'], 
                                    length_scale=3)

# vector_db = KnowledgeBaseManager(qdrant_url=os.getenv('QDRANT_URL'),
#                                  qdrant_api_key=os.getenv('QDRANT_API_KEY'),
#                                  openai_api_key=os.getenv('OPENAI_API_KEY'))

async def transcribe_deepgram(file_path):
    async with dg_client.transcription.prerecorded({'buffer': file_path}, {'punctuate': True}) as response:
        return response

def get_audio_from_url(url):
    # Download the audio file
    response = requests.get(url)

    # Check if the download was successful
    response.raise_for_status()

    # Return the binary content
    return response.content

@app.get('/')
def yoyo():
    return "site is working!!"

@app.get("/list-riva-models")
def list_available_riva_models():
    return list(asr_models.keys())

@app.get("/list-piper-models")
def list_available_piper_models():
    return list(piper_models_config.keys())

@app.post("/transcribe-web-riva")
async def transcribe_audio_web_riva(audio: UploadFile = File(...), model_name: str = Form(...), Authorization: str = Header(...), word_timestamps:bool = Form(0), boosted_lm_words:str = Form('[]'), boosted_lm_score:int = Form(20)):
    sessionid = Authorization.replace('Bearer ', '')
    if model_name not in asr_models.keys():
        raise HTTPException(status_code=400, detail="invalid model name")
    import ast
    boosted_lm_words = ast.literal_eval(boosted_lm_words)

    audio = await audio.read()
    audio_file = await preprocess_audio(audio)

    transcript = ""

    try:
        auth_nlp = riva.client.Auth(uri=os.getenv('NLP_MODEL_URI'))
        model_info = asr_models[model_name]
        riva_nlp = riva.client.NLPService(auth_nlp)

        final_transcript, timestamps, duration_seconds = get_transcript(audio_file, model_info, boosted_lm_words, boosted_lm_score, word_timestamps)
        
        # transcript = riva.client.nlp.extract_most_probable_transformed_text(
        #     riva_nlp.punctuate_text(
        #         input_strings=final_transcript, model_name="riva-punctuation-en-US", language_code='en-US'
        #     )
        # )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "transcript" : final_transcript,
        "duration_seconds": duration_seconds,
        "timestamps": timestamps,
        "language_code": model_name.split('-')[0]
    }

@app.post("/transcribe-diarize-riva")
async def transcribe_diarize_riva(audio: UploadFile = File(...), model_name: str = Form(...), Authorization: str = Header(...), word_timestamps:bool = Form(0), boosted_lm_words:str = Form('[]'), boosted_lm_score:int = Form(20), max_speakers:int = Form(2)):
    sessionid = Authorization.replace('Bearer ', '')
    if model_name not in asr_models.keys():
        raise HTTPException(status_code=400, detail="invalid model name")
    import ast
    boosted_lm_words = ast.literal_eval(boosted_lm_words)

    audio_data = await audio.read()
    audio_file = await preprocess_audio(audio_data)

    transcript = ""

    try:
        auth_nlp = riva.client.Auth(uri=os.getenv('NLP_MODEL_URI'))
        model_info = asr_models[model_name]
        riva_nlp = riva.client.NLPService(auth_nlp)
        final_transcript, timestamps, duration_seconds = get_transcript(audio_file, model_info, boosted_lm_words, boosted_lm_score, word_timestamps, True, max_speakers)
        diarize_output = transform_riva_output(timestamps)

        # transcript = riva.client.nlp.extract_most_probable_transformed_text(
        #     riva_nlp.punctuate_text(
        #         input_strings=final_transcript, model_name="riva-punctuation-en-US", language_code='en-US'
        #     )
        # )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "transcript" : final_transcript,
        "diarize_output": diarize_output,
        "duration_seconds": duration_seconds,
        "timestamps": timestamps,
        "language_code": model_name.split('-')[0]
    }

@app.post("/transcribe-json-input-riva")
async def transcribe_json_input_riva(json_file: UploadFile = File(...)):
    """
    Process a JSON file containing transcription requests and return transcription results.
    
    Args:
        json_file: Upload file containing JSON data with transcription parameters
        
    Returns:
        Dictionary containing transcript, duration, and optional timestamps
    """
    try:
        # Parse JSON input
        message = await json_file.read()
        json_data = json.loads(message)
        output = []

        for data in json_data:
            print(data)
            # Extract request parameters
            model_name = data['model_name']
            audio_file_path = data['audio_file_path']
            word_timestamps = data['word_timestamps']
            boosted_lm_words = data['boosted_lm_words']
            boosted_lm_score = data['boosted_lm_score']

            # Validate model name
            if model_name not in asr_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model name in JSON: {data}"
                )

            audio_message = get_audio_from_url(audio_file_path)

            # Convert audio to WAV format
            cmd = 'ffmpeg -i - -ac 1 -ar 16000 -f wav -'
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            audio_data, error = await proc.communicate(audio_message)

            if error:
                raise HTTPException(status_code=400, detail="Error converting audio file")
            try:
                # Initialize Riva services
                auth_nlp = riva.client.Auth(uri=os.getenv('NLP_MODEL_URI'))
                model_info = asr_models[model_name]
                auth = riva.client.Auth(uri=model_info['uri'])
                riva_asr = riva.client.ASRService(auth)
                riva_nlp = riva.client.NLPService(auth_nlp)

                # Configure Riva ASR
                riva_config = riva.client.RecognitionConfig()
                if boosted_lm_words:
                    riva.client.add_word_boosting_to_config(
                        riva_config,
                        boosted_lm_words,
                        boosted_lm_score
                    )

                riva_config.max_alternatives = 1
                riva_config.enable_automatic_punctuation = True
                riva_config.audio_channel_count = 1
                riva_config.enable_word_time_offsets = word_timestamps
                riva_config.model = model_info['model']

                if 'language_code' in model_info:
                    riva_config.language_code = model_info['language_code']

                # Perform transcription
                response = riva_asr.offline_recognize(audio_data, riva_config)

                # Process transcription results
                transcripts = [result.alternatives[0].transcript for result in response.results]
                duration_seconds = sum(result.audio_processed for result in response.results)
                final_transcript = " ".join(transcripts)

                # Apply punctuation
                transcript = riva.client.nlp.extract_most_probable_transformed_text(
                    riva_nlp.punctuate_text(
                        input_strings=final_transcript,
                        model_name="riva-punctuation-en-US",
                        language_code='en-US'
                    )
                )

                # Process timestamps if requested
                timestamps = []
                if word_timestamps:
                    for result in response.results:
                        timestamps.extend(list(result.alternatives[0].words))
                    timestamps = [MessageToDict(timestamp) for timestamp in timestamps]

                # Add results to output
                output.append({
                    "transcript": transcript,
                    "duration_seconds": duration_seconds,
                    "timestamps": timestamps
                })

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return output

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate-text")
async def translate_text(text: str = Form(...), source_language: str = Form('en'), target_language: str = Form(...)):
    if source_language==target_language:
        return {
            "translated_text": text
        }

    auth = riva.client.Auth(uri=os.getenv('MT_MODEL_URI'))
    riva_nmt_client = riva.client.NeuralMachineTranslationClient(auth)
    parts = nmt_large_text_split(text)
    response = riva_nmt_client.translate(parts, "megatronnmt_any_any_1b", source_language, target_language)

    translated_text = " ".join([translation.text for translation in response.translations])

    return {
        "translated_text": translated_text
    }

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

class LLMSummarizerRequest(BaseModel):
    content: str
    model_name: str
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

@app.post("/llm_text_summarizer", response_model=LLMResponse)
async def llm_text_summarizer(request: LLMSummarizerRequest):
    content = request.content
    model_name = request.model_name
    instruction = request.instruction

    print("Input Text:", content)
    print("Model Name:", model_name)
    print("Instruction:", instruction)

    if model_name == "openai":
        input_text = clean_tags(content)
        text, input_tokens, output_tokens = get_openai_response(input_text, instruction, os.getenv('OPENAI_API_KEY'))
        return {"response": text, "input_text": content, "input_tokens": input_tokens, "output_tokens": output_tokens}
    else:
        return {"response": "Model not found", "input_text": content, "input_tokens": 0, "output_tokens": 0}

def extract_metadata(text: str) -> dict:
    """Extract metadata tags from input text.
    
    Args:
        text: Input text containing metadata tags
        
    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        'emotion': None,
        'age': None,
        'intent': None,
        'gender': None,
        'dialect': None
    }
    
    # Extract emotion
    emotion_match = re.search(r'EMOTION_(\w+)', text)
    if emotion_match:
        metadata['emotion'] = emotion_match.group(1)
    
    # Extract age
    age_match = re.search(r'AGE_(\d+_\d+)', text)
    if age_match:
        metadata['age'] = age_match.group(1)
    
    # Extract intent
    intent_match = re.search(r'INTENT_(\w+)', text)
    if intent_match:
        metadata['intent'] = intent_match.group(1)
    
    # Extract gender
    gender_match = re.search(r'GENDER_(\w+)', text)
    if gender_match:
        metadata['gender'] = gender_match.group(1)
    
    # Extract dialect
    dialect_match = re.search(r'DIALECT_(\w+)', text)
    if dialect_match:
        metadata['dialect'] = dialect_match.group(1)
    
    return metadata

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
    print("Role:", role)
    
    print("Search Engine:", searchengine)
    print("Instruction:", system_instruction)
    print("History:", conversation_history)
    
    # Extract metadata from content
    metadata = extract_metadata(content)
    
    # Use provided emotion or extract from metadata
    emotion = emotion.replace("EMOTION_", "") if emotion else metadata['emotion']
    print("Emotion:", emotion)
    
    # Build detailed instruction based on metadata
    instruction_parts = []
    
    if role:
        instruction_parts.append(f"Act as a {role}")
    
    # Add context about the user without explicitly stating it
    context_parts = []
    if emotion:
        context_parts.append(f"respond in a way that acknowledges their {emotion} state")
    
    if metadata['age']:
        context_parts.append(f"use language appropriate for someone in their {metadata['age']} age range")
    
    if metadata['gender']:
        context_parts.append(f"be mindful of gender-specific considerations")
    
    if metadata['dialect']:
        context_parts.append(f"adapt your language style to match {metadata['dialect']} dialect patterns")
    
    if metadata['intent']:
        context_parts.append(f"focus on addressing their {metadata['intent']} needs")
    
    if context_parts:
        instruction_parts.append("while " + " and ".join(context_parts))
    
    detailed_instruction = system_instruction if system_instruction else " and ".join(instruction_parts) + ". Provide a natural, empathetic response without explicitly mentioning these contextual factors."
    
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
        input_text = clean_tags(content)
        if emotion:
            input_text += f' {{emotionalstate: {emotion}}}'
        text, input_tokens, output_tokens = get_openai_response(input_text, detailed_instruction, os.getenv('OPENAI_API_KEY'), conversation_history)
        return {"response": text, "input_text": content, "input_tokens": input_tokens, "output_tokens": output_tokens}
    else:
        response = llm_model_tensorrt.generate_response([content], instructions=detailed_instruction, history=conversation_history, role=role)
        text = response[0]
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
                                 input_file: UploadFile = File(...),
                                 audio_model_name:str = Form("en-NER")):

    print("URL:", url)
    print("Input Text:", content)
    print("Model Name:", model_name)
    print("Emotion:", emotion)
    print("Search Engine:", searchengine)
    print("Instruction:", system_instruction)
    print("History:", conversation_history)
    print("Input File:", input_file)
    
    
    detailed_instruction = system_instruction if system_instruction else f"Considering your role as a {role} and understanding that the speaker is feeling {emotion}, provide an empathetic and context-aware response"
    
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
    filename = secure_filename(input_file.filename)
    file_type = filename.split('.')[-1]

    if DEV_MODE:
        if file_type == 'pdf':
            save_directory = '/root/webaudio'
            os.makedirs(save_directory, exist_ok=True)
            file_path = os.path.join(save_directory, filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(input_file.file, buffer)
            print(f"File saved at {file_path}")
            input_text = clean_tags(content) + f' {{emotionalstate: {emotion}}}'
            # if model_name == 'news_llm':
            #     text = news_llm.generate_rag_response(embeddings, input_text, pdf=file_path, instruction=detailed_instruction)
            # else:
            text = get_rag_response(embeddings, input_text, os.getenv('OPENAI_API_KEY'), pdf=file_path, instruction=detailed_instruction)
            return {"response": text, "input_text": content, "input_tokens": 0, "output_tokens": 0}
        elif file_type in ['mp3', 'wav']:
            input_file = await input_file.read()
            audio_file = await preprocess_audio(input_file)
            transcript = ""

            model_info = asr_models[audio_model_name]
            transcript, token_timestamps, duration = get_transcript(audio_file, model_info, [], 0, False)


            # update input_text or detailed_instruction
            input_text = clean_tags(content) + f' {{emotionalstate: {emotion}}}'
            print(transcript)
            detailed_instruction += " considering the provided audio context with the following transcript with emotion and entities \n" + transcript
            # if model_name == 'news_llm':
            #     text = news_llm.generate_response(input_text, detailed_instruction, conversation_history)
            #     input_tokens, output_tokens = (0,0)
            # else:
            #     text, input_tokens, output_tokens = get_openai_response(input_text, detailed_instruction, conversation_history)
            text, input_tokens, output_tokens = get_openai_response(input_text, detailed_instruction, os.getenv('OPENAI_API_KEY'), conversation_history)
            return {"response": text, "input_text": content, "input_tokens": input_tokens, "output_tokens": output_tokens}
        elif file_type in ['jpeg', 'jpg', 'png']:
            message = await input_file.read()
            input_text = clean_tags(content) + f' {{emotionalstate: {emotion}}}'
            loop = asyncio.get_event_loop()
            caption = await loop.run_in_executor(executor, blip_infer, blip_processor, vision_model_sess, text_model_sess, BytesIO(message))
            print(caption)
            detailed_instruction += " considering the provided image context with the following image caption: {caption}".format(caption=caption)
            # if model_name == 'news_llm':
            #     text = news_llm.generate_response(input_text, detailed_instruction, conversation_history)
            #     input_tokens, output_tokens = (0,0)
            # else:
            #     text, input_tokens, output_tokens = get_openai_response(input_text, detailed_instruction, conversation_history)
            text, input_tokens, output_tokens = get_openai_response(input_text, detailed_instruction, os.getenv('OPENAI_API_KEY'), conversation_history)
            return {"response": text, "input_text": content, "input_tokens": input_tokens, "output_tokens": output_tokens}
        else:
            return 'file format not supported'

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
            test_image = mutlimodal_runner.load_test_image(multimodal_llm_hf_dir, url)
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
                text, input_tokens, output_tokens = get_openai_response(input_text, system_instruction, os.getenv('OPENAI_API_KEY'), conversation_history)
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
        
        rag_response = get_rag_response([url], query, token=os.getenv('OPENAI_API_KEY'))
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
async def generate_audio_xtts(model_name: str = Form("piper"), text_to_convert: str = Form(...), language: str = Form('en-US'), output_filename: str = Form(...), ref_file: UploadFile = File(None)):
    try:
        print("TTS Model Name", model_name, language)
        
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
                raise HTTPException(status_code=400, detail="need ref file for voice cloning")
            audio_content = xtts_model.infer(text= text_to_convert, language=language, file_path=output_filename, speaker_wav_file_path=ref_file_path)
            
            duration_seconds = AudioSegment(audio_content).duration_seconds
            return Response(content=audio_content, media_type="audio/mpeg", headers={"X-Duration": str(duration_seconds)})
        elif model_name == "piper":
            text_to_convert = clean_text_for_piper(text_to_convert)
            if language not in piper_models.keys():
                raise HTTPException(status_code=400, detail="language isn't supported yet")
            audio_content = piper_models[language].synthesize(text_to_convert)
            duration_seconds = AudioSegment(audio_content).duration_seconds
            return Response(content=audio_content, media_type="audio/mpeg", headers={"X-Duration": str(duration_seconds)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/get_audio_language")
async def get_audio_language(audio: UploadFile = File(...)):
    audio = await audio.read()
    audio = await preprocess_audio(audio)

    first_few_seconds_audio = await extract_first_n_seconds_of_audio(audio, 10)
    return infer_ambernet_onnx(ort_session_ambernet, filterbank_featurizer, labels, first_few_seconds_audio)

@app.post("/transcribe_auto_language")
async def get_auto_transcription(audio: UploadFile = File(...), word_timestamps:bool = Form(0)):
    audio_data = await audio.read()
    audio_data = await preprocess_audio(audio_data)

    first_few_seconds_audio = await extract_first_n_seconds_of_audio(audio_data, 10)
    lang_id = infer_ambernet_onnx(ort_session_ambernet, filterbank_featurizer, labels, first_few_seconds_audio)

    if lang_id not in lang_to_model_map.keys():
        raise HTTPException(status_code=400, detail=lang_id+" not supported yet")

    model_name = lang_to_model_map[lang_id]
    try:
        model_info = asr_models[model_name]
        final_transcript, timestamps, duration_seconds = get_transcript(audio_data, model_info, [], 20, word_timestamps)
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "transcript" : final_transcript,
        "duration_seconds": duration_seconds,
        "timestamps": timestamps,
        "language_code": model_name.split('-')[0]
    }


# Start of RAG related endpoints for text summarization with Qdrant
@app.post("/create_knowledge_base")
async def create_knowledge_base(collection_name: str = Form(...)):
    try:
        intitalized = vector_db.initialize_collection(collection_name=collection_name)
        if intitalized:
            return {"message": "Collection created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_files_to_knowledge_base")
async def add_files_to_knowledge_base(
    files: list[UploadFile] = File(...),
    collection_name: str = Form(...)
):
    try:
        # Initialize vector database connection once for all files
        intitalized = vector_db.initialize_collection(
            collection_name=collection_name)
        
        if intitalized:
            results = []
            for file in files:
                try:
                    # Read file content directly into memory
                    file_content = await file.read()

                    # Create a BytesIO object to work with the file content in memory
                    file_stream = BytesIO(file_content)

                    # Process the file and add it to the knowledge base
                    loaded = vector_db.process_byteio_file(file_stream, file.filename)

                    if loaded:
                        results.append({
                            "filename": file.filename,
                            "status": "success"
                        })
                    else:
                        results.append({
                            "filename": file.filename,
                            "status": "failed",
                        })

                except Exception as file_error:
                    results.append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": str(file_error)
                    })

            return {
                "message": "Files processing completed",
                "results": results
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/query_llm_with_knowledge_base")
async def query_llm_with_knowledge_base(collection_name: str = Form(...), query: str = Form(...), model_name: str = Form("gpt-3.5-turbo")):
    try:
        # Initialize vector database connection once for all files
        intitalized = vector_db.initialize_collection(
            collection_name=collection_name)

        if intitalized:
            # Query the knowledge base
            response = vector_db.query_documents(
                query=query,
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model_name=model_name,
                temperature=0.0,
                num_documents=3
            )

            return {
                "response": response
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class StandaloneApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == '__main__':
    # from uvicorn import Config, Server

    # server_config = Config(app, host='0.0.0.0', port=5000, workers=3)
    # server = Server(server_config)
    # server.run()
    options = {
        "bind": "0.0.0.0:5000",
        "workers": 2,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "preload_app": True,
    }

    StandaloneApplication(app, options).run()

