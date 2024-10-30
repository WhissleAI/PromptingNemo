# Wellness Companion Chatbot

This project implements a wellness companion chatbot using FastAPI, OpenAI's GPT-3.5-turbo model, and Sentence Transformers for natural language processing. The chatbot provides personalized recommendations based on user queries, rephrasing suggestions in an engaging manner and linking to relevant YouTube videos.

## Features

- **FastAPI** for serving the chatbot as a web service.
- **OpenAI GPT-3.5-turbo** for generating natural language responses.
- **Sentence Transformers** for finding the most relevant response using similarity search.
- **FAISS** for efficient similarity search on dense vector embeddings.
- **Pandas** for data handling and processing.

## Installation

### Prerequisites

- Python 3.7+
- An OpenAI API key

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/wellness-companion-chatbot.git
    cd wellness-companion-chatbot
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key:

    Replace `'your-openai-api-key-here'` in `app.py` with your actual OpenAI API key.

    Alternatively, you can export it as an environment variable:

    ```bash
    export OPENAI_API_KEY='your-openai-api-key-here'
    ```

4. Run the FastAPI server:

    ```bash
    uvicorn app:app --reload
    ```

5. The server will start at `http://127.0.0.1:8000`. You can now send POST requests to the `/query/` endpoint.

## Usage

You can interact with the chatbot by sending a POST request to the `/query/` endpoint with a JSON body containing your query. Example:

```bash
curl -X POST "http://127.0.0.1:8000/query/" -H "Content-Type: application/json" -d '{"query": "I am feeling stressed and anxious."}'
```

