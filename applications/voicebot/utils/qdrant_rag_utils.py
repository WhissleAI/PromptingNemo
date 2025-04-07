from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import PyPDF2
from langchain.schema import Document
from io import BytesIO


def get_timestamp():
    """Helper function to get current timestamp for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class KnowledgeBaseManager:
    """A class to manage document processing and querying using Qdrant vector database."""

    def __init__(
        self,
        qdrant_host: str = None,
        qdrant_port: int = None,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        openai_api_key: str = None,
        embedding_model: str = "text-embedding-ada-002"
    ):
        """
        Initialize VectorDB manager with Qdrant and embedding configurations.
        
        Args:
            qdrant_host: Hostname for Qdrant server
            qdrant_port: Port number for Qdrant server
            qdrant_url: URL for Qdrant server
            qdrant_api_key: API key for Qdrant
            openai_api_key: API key for OpenAI
            embedding_model: OpenAI embedding model to use
        """
        print(f"[{get_timestamp()}] Initializing KnowledgeBaseManager client")
        self.embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.vector_store = None
        self.collection_name = ''
        self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            if not (self.qdrant_url and self.qdrant_api_key):
                raise ValueError("Qdrant URL and API key are required")

            print(f"[{get_timestamp()}] Connecting to Qdrant at {self.qdrant_url}")
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
        except Exception as e:
            print(
                f"[{get_timestamp()}] ERROR: Failed to initialize collection: {str(e)}")
            raise

    def initialize_collection(self, collection_name: str = 'documents') -> bool:
        """Initialize or connect to the Qdrant collection."""
        self.collection_name = collection_name
        try:
            collections = self.client.get_collections().collections
            exists = any(collection.name == self.collection_name for collection in collections)

            if exists:
                print(f"[{get_timestamp()}] Connecting to existing collection: {self.collection_name}")
                self.vector_store = QdrantVectorStore.from_existing_collection(
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    url=self.qdrant_url,
                    prefer_grpc=True,
                    api_key=self.qdrant_api_key,
                )
                return True
            else:
                print(f"[{get_timestamp()}] Creating new collection: {self.collection_name}")
                self.vector_store = self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536, distance=Distance.COSINE)
                )
                return True
        except Exception as e:
            print(f"[{get_timestamp()}] ERROR: Failed to initialize collection: {str(e)}")
            raise

    def process_file(
        self,
        file_path: str,
        chunk_size: int = 1000,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Process a single file and add it to the vector database.
        
        Args:
            file_path: Path to the document
            chunk_size: Size of text chunks for splitting
            metadata: Additional metadata to store with document
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        print(f"[{get_timestamp()}] Processing file: {file_path}")
        try:
            doc = self._load_document(file_path)
            chunks = self._split_documents(doc, chunk_size)

            if metadata:
                print(f"[{get_timestamp()}] Adding metadata to chunks: {metadata}")
                for chunk in chunks:
                    chunk.metadata.update(metadata)
                    chunk.metadata['source'] = file_path

            self.vector_store.add_documents(chunks)
            print(f"[{get_timestamp()}] Successfully processed file: {file_path}")
            return True, f"Successfully processed {file_path}"

        except Exception as e:
            print(f"[{get_timestamp()}] ERROR: Error processing file {file_path}: {str(e)}")
            return False, f"Error processing {file_path}: {str(e)}"

    def _load_document(self, file_path: str) -> List[Document]:
        """Load a document from file path."""
        print(f"[{get_timestamp()}] Loading document: {file_path}")
        extension = os.path.splitext(file_path)[1].lower()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            docs = []
            if extension == '.pdf':
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        text = reader.pages[page_num].extract_text()
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": file_path, "page": page_num + 1}
                        ))
                print(f"[{get_timestamp()}] Loaded PDF with {len(docs)} pages")

            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    docs = [Document(
                        page_content=text,
                        metadata={"source": file_path}
                    )]
                print(f"[{get_timestamp()}] Loaded text file: {len(text)} characters")

            else:
                raise ValueError(f"Unsupported file type: {extension}")

            return docs

        except Exception as e:
            print(f"[{get_timestamp()}] ERROR: Error loading document {file_path}: {str(e)}")
            raise

    def process_byteio_file(
        self,
        file_obj: BytesIO,
        filename: str,
        chunk_size: int = 1000
    ) -> bool:
        """Process a file from a BytesIO object."""
        print(f"[{get_timestamp()}] Processing BytesIO file: {filename}")
        try:
            file_obj.seek(0)
            extension = os.path.splitext(filename)[1].lower()

            if extension not in ['.pdf', '.txt']:
                raise ValueError(f"Unsupported file type: {extension}")

            docs = []
            if extension == '.pdf':
                reader = PyPDF2.PdfReader(file_obj)
                for page_num in range(len(reader.pages)):
                    text = reader.pages[page_num].extract_text()
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1}
                    ))
                print(f"[{get_timestamp()}] Processed PDF with {len(docs)} pages")

            elif extension == '.txt':
                text = file_obj.read().decode('utf-8')
                docs = [Document(
                    page_content=text,
                    metadata={"source": filename}
                )]
                print(f"[{get_timestamp()}] Processed text file: {len(text)} characters")

            chunks = self._split_documents(docs, chunk_size)
            self.vector_store.add_documents(chunks)
            print(f"[{get_timestamp()}] Successfully added documents to vector store")
            return True

        except Exception as e:
            print(f"[{get_timestamp()}] ERROR: Error processing BytesIO file {filename}: {str(e)}")
            raise

    def _split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000
    ) -> List[Document]:
        """Split documents into smaller chunks."""
        print(f"[{get_timestamp()}] Splitting documents into chunks of size {chunk_size}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[{get_timestamp()}] Created {len(chunks)} chunks from documents")
        return chunks

    def query_documents(
        self,
        query: str,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 500,
        num_documents: int = 3
    ) -> Dict[str, Any]:
        """
        Query the vector database using RAG.
        
        Args:
            query: User's question
            openai_api_key: OpenAI API key
            model_name: Name of the LLM model to use
            temperature: Temperature for the LLM
            max_tokens: Maximum tokens in the response
            num_documents: Number of documents to retrieve
        """
        print(f"[{get_timestamp()}] Querying documents with model {model_name}")
        try:
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=openai_api_key
            )

            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": num_documents}
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            print(f"[{get_timestamp()}] Executing query: {query}")
            response = qa_chain({'query': query})

            result = {
                'answer': response['result'],
                'sources': [{
                    'source': doc.metadata.get('source', 'Unknown'),
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in response['source_documents']]
            }

            print(f"[{get_timestamp()}] Query completed successfully")
            return result

        except Exception as e:
            print(f"[{get_timestamp()}] ERROR: Error executing query: {str(e)}")
            raise

    def delete_collection(self) -> bool:
        """Delete the entire collection from Qdrant."""
        print(f"[{get_timestamp()}] Attempting to delete collection: {self.collection_name}")
        try:
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
            client.delete_collection(self.collection_name)
            print(f"[{get_timestamp()}] Successfully deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"[{get_timestamp()}] ERROR: Error deleting collection: {str(e)}")
            return False


# if __name__ == "__main__":
#     load_dotenv()

#     try:
#         print(f"[{get_timestamp()}] Initializing KnowledgeBaseManager")
#         vector_db = KnowledgeBaseManager(
#             qdrant_url=os.getenv("QDRANT_HOST"),
#             qdrant_api_key=os.getenv("QDRANT_API_KEY"),
#             openai_api_key=os.getenv("OPENAI_API_KEY"),
#             collection_name="test_pdf_collection"
#         )

#         files = [
#             "./data/pdf1.pdf",
#             "/external1/nihars/workspace/vectordb-testing/data/pdf2.pdf",
#             "/external1/nihars/workspace/vectordb-testing/data/pdf3.pdf",
#             "/external1/nihars/workspace/vectordb-testing/data/pdf4.pdf"
#         ]
#         metadata = {"organization": "example_org"}

#         print(f"[{get_timestamp()}] Processing files")
#         results = vector_db.process_files(
#             file_paths=files,
#             chunk_size=1000,
#             metadata=metadata
#         )

#         print(f"[{get_timestamp()}] Executing test query")
#         response = vector_db.query_documents(
#             query="What are the letters about? And whom are they written to?",
#             openai_api_key=os.getenv("OPENAI_API_KEY"),
#             model_name="gpt-4",
#             temperature=0.0,
#             num_documents=3
#         )

#         print(f"\nAnswer: {response['answer']}")
#         print("\nSources:")
#         for source in response['sources']:
#             print(f"- {source['source']}: {source['content'][:100]}...")

#     except Exception as e:
#         print(f"[{get_timestamp()}] ERROR: Main script error: {str(e)}")
