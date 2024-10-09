from langchain_community.document_loaders import UnstructuredURLLoader, PyMuPDFLoader

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders import Document
import openai
# import pandas as pd


# def get_rag_response_from_csv(embeddings, csv_path, query, instruction=None):
#     openai.api_key = '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'
    
#     # Load CSV file
#     df = pd.read_csv(csv_path)
    
#     # Extract document texts and metadata
#     documents = df['document_text'].tolist()
#     metadata = df['metadata'].tolist()
    
#     # Create Document objects with metadata
#     data = [Document(text=text, metadata={'num': i, 'info': meta}) for i, (text, meta) in enumerate(zip(documents, metadata))]
    
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     data = text_splitter.split_documents(data)
#     print("Length of data:", len(data))

#     vectorstore = FAISS.from_documents(data, embedding=embeddings)

#     llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", openai_api_key=openai.api_key)
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(),
#             memory=memory
#             )
    
#     # Combine custom instruction with the query
#     if instruction:
#         query = f"{instruction}\n\n{query}"
    
#     result = conversation_chain.invoke({"question": query})
#     answer = result["answer"]
#     # Retrieve document metadata used for the answer
#     used_documents = result["source_documents"]
#     document_numbers = [doc.metadata['num'] for doc in used_documents]
    
#     return answer, document_numbers



def get_rag_response(embeddings, query, urls=None, pdf=None, instruction=None):
    openai.api_key='sk-proj-wBhxVeSmc5c9wq0MccFNT3BlbkFJPnPgz351rUnyoyLziIRu'
    
    if urls:
        loader = UnstructuredURLLoader(urls=urls)
    elif pdf:
        loader = PyMuPDFLoader(pdf)
    data = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    data = text_splitter.split_documents(data)
    print("Length of data:", len(data))

    vectorstore = FAISS.from_documents(data, embedding=embeddings)  

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", openai_api_key=openai.api_key)
    memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=memory
            )
    
    # Combine custom instruction with the query
    if instruction:
        query = f"{instruction}\n\n{query}"
    
    result = conversation_chain.invoke({"question": query})
    answer = result["answer"]
    return answer