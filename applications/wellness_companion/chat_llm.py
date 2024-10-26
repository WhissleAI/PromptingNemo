import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Manually set the correct column names
columns = ['Question', 'Answers', 'Recommendation', 'LINK', 'KeyWords']

# Read the TSV file, assuming it's tab-separated, and assign column names
data = pd.read_csv('renew-u.tsv', sep='\t', header=None, names=columns)

# Strip any extra spaces from column names (just in case)
data.columns = data.columns.str.strip()

# Combine context for embeddings by merging 'Answers' and 'Recommendation' columns
data['combined_context'] = data['Answers'] + " " + data['Recommendation']

# Print the first few rows to verify the data is loaded correctly
print(data.head())

# Load a smaller model and specify that it should run on the CPU
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')

# Generate embeddings using the CPU
embeddings = model.encode(data['combined_context'].tolist(), device='cpu')

# Create a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Function to query the chatbot
def query_chatbot(query):
    query_embedding = model.encode([query], device='cpu')
    D, I = index.search(query_embedding, k=1)  # Search for the closest match
    result = data.iloc[I[0][0]]
    return {
        'description': result['combined_context'],
        'youtube_link': result['LINK'],
        'recommendation': result['Recommendation']
    }

# Example query to test the chatbot
response = query_chatbot("I'm feeling anxious and stressed")
print(response)
