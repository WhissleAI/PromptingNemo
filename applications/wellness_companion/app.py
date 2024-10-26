from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = 'OPENAI_API_KEY'

# Initialize FastAPI
app = FastAPI()

# Mount the static files directory to serve the UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load data and model only once at startup
columns = ['Question', 'Answers', 'Recommendation', 'LINK', 'KeyWords']
data = pd.read_csv('renew-u.tsv', sep='\t', header=None, names=columns)
data.columns = data.columns.str.strip()
data['combined_context'] = data['Answers'] + " " + data['Recommendation']

model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
embeddings = model.encode(data['combined_context'].tolist(), device='cpu')
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Define a Pydantic model for the incoming request
class QueryRequest(BaseModel):
    query: str

# Function to generate a response using ChatGPT (gpt-3.5-turbo)
def generate_chatgpt_response(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful wellness companion."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

# Function to handle the query and get a ChatGPT-generated response
def query_chatbot(query: str) -> dict:
    query_embedding = model.encode([query], device='cpu')
    D, I = index.search(query_embedding, k=1)  # Search for the closest match
    result = data.iloc[I[0][0]]
    

    # Create a prompt for ChatGPT to rephrase the recommendation
    prompt = (
        f"You asked: {query}\n\n"
        f"Here is a situation related to your query:\n\n"
        f"{result['Answers']}\n\n"
        f"Please rephrase the following recommendation to make it more engaging. not unnecessay words than rephrasing the recommendation\n\n"
        f"{result['Recommendation']}\n\n"
        f"You can also watch this video for more guidance: {result['LINK']}"
    )

    # Generate the ChatGPT response
    chatgpt_response = generate_chatgpt_response(prompt)

    return {
        'chatgpt_response': chatgpt_response,
        'youtube_link': result['LINK'],
        'recommendation': result['Recommendation']
    }

def query_chatbot_multiple(query: str) -> dict:
    # Get the query embedding
    query_embedding = model.encode([query], device='cpu')
    
    # Search for the top 3 closest matches
    D, I = index.search(query_embedding, k=3)
    
    # Gather the recommendations and YouTube links from the top 3 matches
    recommendations = []
    youtube_links = []
    for i in range(3):
        result = data.iloc[I[0][i]]
        recommendations.append(result['Recommendation'])
        youtube_links.append(result['LINK'])
    
    # Combine all recommendations into a single prompt
    combined_recommendations = "\n\n".join(recommendations)
    prompt = (
        f"You asked: {query}\n\n"
        f"Here are some situations related to your query:\n\n"
        f"{combined_recommendations}\n\n"
        f"Please rephrase the above recommendations to make them more engaging and concise."
    )
    
    # Generate the ChatGPT response
    chatgpt_response = generate_chatgpt_response(prompt)
    
    return {
        'chatgpt_response': chatgpt_response,
        'youtube_links': youtube_links
    }
    
# FastAPI route to handle POST requests with user queries
@app.post("/query/")
async def handle_query(request: QueryRequest):
    try:
        response = query_chatbot_multiple(request.query)
        print(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the UI as the root endpoint
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)
