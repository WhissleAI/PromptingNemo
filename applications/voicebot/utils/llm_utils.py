from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import requests
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredURLLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from abc import ABC, abstractmethod
import time

# Define BaseLLM first
class BaseLLM(ABC):
    @abstractmethod
    def _generate(self, prompt, max_length):
        pass

    @abstractmethod
    def _llm_type(self):
        pass

class HFLanguageModel:
    def __init__(self, model_name_or_path):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            device=device if device != 'mps' else -1,
            trust_remote_code=True,
            max_new_tokens=150
        )
        self.rag_pipeline = HuggingFacePipeline(pipeline=self.pipeline)

    def generate_response(self, input_text, instruction, conversation_history):
        messages = [{"role": "system", "content": instruction}]

        for entry in conversation_history:
            if entry.get("query"):
                messages.append({"role": "user", "content": entry["query"]})
            if entry.get("answer"):
                messages.append({"role": "assistant", "content": entry["answer"]})

        messages.append({"role": "user", "content": input_text})

        generation_args = { 
            "max_new_tokens": 500, 
            "return_full_text": False, 
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": self.llm_tokenizer.pad_token_id,
        } 
        
        try:
            response = self.pipeline(messages, **generation_args)
            return response[0]['generated_text']
        except Exception as e:
            print(f"Generation error: {e}")
            return f"An error occurred during generation: {str(e)}"
    
    def generate_rag_response(self, embeddings, query, urls=None, pdf=None, instruction=None):
        try:
            if urls:
                loader = UnstructuredURLLoader(urls=urls)
            elif pdf:
                loader = PyMuPDFLoader(pdf)
            else:
                return "No source provided for RAG"
                
            data = loader.load()
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            data = text_splitter.split_documents(data)
            print("Length of data:", len(data))

            vectorstore = FAISS.from_documents(data, embedding=embeddings)  

            memory = ConversationBufferMemory(
                memory_key='chat_history', return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.rag_pipeline,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    memory=memory
            )
            
            if instruction:
                query = f"{instruction}\n\n{query}"

            result = conversation_chain.invoke({"question": query})
            return result["answer"]
        except Exception as e:
            print(f"RAG error: {e}")
            return f"An error occurred during RAG processing: {str(e)}"

class HuggingFaceAPI(BaseLLM):
    def __init__(self, model_id, api_token):
        self.model_id = model_id
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}"
        }
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    def _generate(self, user_input, max_length=100, retries=5, wait_time=30):
        prompt_template = f"""
        User: {user_input}
        Assistant: 
        """
        
        payload = {
            "inputs": prompt_template.strip(),
            "parameters": {
                "max_length": max_length,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                
                print(response.text)
                
                response_json = response.json()
                if isinstance(response_json, list):
                    generated_text = response_json[0].get('generated_text', '').strip()
                else:
                    generated_text = response_json.get('generated_text', '').strip()
                
                assistant_reply_start = generated_text.find("Assistant:")
                if assistant_reply_start != -1:
                    generated_text = generated_text[assistant_reply_start + len("Assistant:"):].strip()
                
                return generated_text

            except requests.exceptions.HTTPError as errh:
                if response.status_code == 503 and 'loading' in response.text.lower():
                    print(f"Model is loading, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(wait_time)
                else:
                    print(f"HTTP Error: {errh}\nResponse Content: {response.text}")
                    break
            except Exception as e:
                print(f"Error during API call: {str(e)}")
                if attempt == retries - 1:
                    return f"Error after {retries} attempts: {str(e)}"
                time.sleep(wait_time)
                
        return "Failed to generate response after multiple attempts"

    def _llm_type(self):
        return "Hugging Face API"