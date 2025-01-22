"""Data processing module for ingesting and processing knowledge base content."""
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import CrateDB

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_endpoint=config.embeddings.endpoint
        )
        self.vector_store = CrateDB(
            connection_string=config.vector_store.connection_string,
            embeddings=self.embeddings,
            table_name="documents"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    async def process_directory(self, directory_path: str) -> None:
        """Process all text files in a directory."""
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        texts = self.text_splitter.split_documents(documents)
        
        # Store in vector database
        self.vector_store.add_documents(texts)
    
    async def process_text(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """Process a single text string."""
        texts = self.text_splitter.split_text(text)
        self.vector_store.add_texts(texts, metadatas=[metadata] * len(texts) if metadata else None)
    
    def search(self, query: str, k: int = 4) -> List[str]:
        """Search for relevant documents."""
        return self.vector_store.similarity_search(query, k=k)
