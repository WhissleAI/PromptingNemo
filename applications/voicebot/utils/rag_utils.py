from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai

def get_rag_response(embeddings, urls, query, instruction=None):
    openai.api_key='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'
    
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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