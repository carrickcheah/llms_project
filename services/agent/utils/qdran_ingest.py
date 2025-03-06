from langchain.schema import Document
from qdrant_client import QdrantClient
import json
from loguru import logger
from typing import Optional, List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import hashlib
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env', 
        env_file_encoding='utf-8'
    )
    url: str
    collection_name: str
    prefer_grpc: bool
    file_path: str
    hf_model: str
    content_payload_key: Optional[str] = None
    deepseek_api_key: str
    deepseek_url: str
    deepseek_model: str

class VectorDBManager:
    """Manages vector database operations with duplicate detection"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.client = QdrantClient(url=config.url, prefer_grpc=config.prefer_grpc)
        
        logger.info("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.hf_model
        )
        
        self.vector_db = None
    
    def generate_document_hash(self, text: str) -> str:
        """Generate a unique hash for a document text to identify duplicates"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def load_documents_from_file(self, file_path: str) -> List[Document]:

        """
        Load documents from file with appropriate error handling
        """
        try:
            # Try reading as JSONL first (JSON Lines - one JSON object per line)
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError:
                            # If line-by-line parsing fails for any line, fall back to standard parsing
                            data = []
                            break
            
            # If JSONL parsing didn't work or didn't find any valid lines, try as single JSON
            if not data:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"File is not valid JSON or JSONL: {str(e)}")
                    # Create a dummy document to avoid crashing
                    return [Document(page_content=f"Error loading documents: {str(e)}")]
            
            # First, let's debug the structure of the JSON file
            if isinstance(data, list) and len(data) > 0:
                sample_item = data[0]
                logger.info(f"Sample data item keys: {list(sample_item.keys())}")
                
                # Try to identify the content field
                content_field = None
                possible_content_fields = ['page_content', 'metadata']
                
                # Check if content_payload_key is specified in config
                if self.config.content_payload_key:
                    possible_content_fields.insert(0, self.config.content_payload_key)
                
                for field in possible_content_fields:
                    if field in sample_item:
                        content_field = field
                        break
                
                if content_field:
                    logger.info(f"Using '{content_field}' as content field")
                    docs = [Document(page_content=item[content_field]) for item in data]

                # After (include metadata)
                if content_field:
                    docs = [
                        Document(
                            page_content=item[content_field],
                            metadata=item.get("metadata", {})  # Add this line
                        ) 
                        for item in data
                    ]
                    
                else:
                    # If we can't identify a specific field, use the entire item as content
                    logger.warning("Could not identify content field, using full JSON objects")
                    docs = [Document(page_content=json.dumps(item)) for item in data]
            elif isinstance(data, dict):
                # If the JSON is a single object or has a different structure
                logger.info(f"Data is a dictionary with keys: {list(data.keys())}")
                
                # If there's an items array or similar
                for key in ['items', 'documents', 'entries', 'data']:
                    if key in data and isinstance(data[key], list):
                        return self.load_documents_from_file_content(data[key])
                
                # Otherwise just use the whole object
                docs = [Document(page_content=json.dumps(data))]
            else:
                logger.warning(f"Unexpected data format: {type(data)}")
                docs = [Document(page_content=json.dumps(data))]
            
            logger.success(f"Loaded {len(docs)} documents from {file_path}")
            return docs
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            # Create a dummy document to avoid crashing
            return [Document(page_content=f"Error loading documents: {str(e)}")]
    
    def load_documents_from_file_content(self, data: List[Dict]) -> List[Document]:
        """Helper to load documents from a list of dictionaries"""
        if not data:
            return []
        
        sample_item = data[0]
        logger.info(f"Sample data item keys: {list(sample_item.keys())}")
        
        # Try to identify the content field
        content_field = None
        possible_content_fields = ['text', 'content', 'page_content', 'body', 'data']
        
        # Check if content_payload_key is specified in config
        if self.config.content_payload_key:
            possible_content_fields.insert(0, self.config.content_payload_key)
        
        for field in possible_content_fields:
            if field in sample_item:
                content_field = field
                break
        
        if content_field:
            logger.info(f"Using '{content_field}' as content field")
            return [Document(page_content=item[content_field]) for item in data]
        else:
            # If we can't identify a specific field, use the entire item as content
            logger.warning("Could not identify content field, using full JSON objects")
            return [Document(page_content=json.dumps(item)) for item in data]
    
    def filter_duplicates(self, documents: List[Document]) -> List[Document]:
        """Filter out documents that already exist in the database based on content hash"""
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.config.collection_name not in collection_names:
            logger.info(f"Collection {self.config.collection_name} doesn't exist yet, no duplicates to filter")
            return documents
        
        # Generate hashes for all input documents
        doc_hashes = {self.generate_document_hash(doc.page_content): doc for doc in documents}
        
        # Get payload from existing documents
        existing_payloads = []
        offset = 0
        limit = 100
        
        while True:
            results = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = results[0]
            if not points:
                break
                
            for point in points:
                if "metadata" in point.payload and "content_hash" in point.payload["metadata"]:
                    existing_payloads.append(point.payload["metadata"]["content_hash"])
            
            offset += limit
            if len(points) < limit:
                break
        
        # Filter out documents that already exist
        unique_docs = []
        for doc_hash, doc in doc_hashes.items():
            if doc_hash not in existing_payloads:
                # Add hash to document metadata
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["content_hash"] = doc_hash
                unique_docs.append(doc)
        
        logger.info(f"Filtered out {len(documents) - len(unique_docs)} duplicate documents")
        return unique_docs
    
    def initialize_or_get_db(self) -> QdrantVectorStore:
        """Initialize or get an existing vector database"""
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.config.collection_name in collection_names:
            logger.success(f"Using existing collection: {self.config.collection_name}")
            # Connect to existing vector store
            self.vector_db = QdrantVectorStore(
                client=self.client,
                collection_name=self.config.collection_name,
                embedding=self.embeddings
            )
        else:
            logger.success(f"Collection {self.config.collection_name} not found. Creating new collection.")
            
            # Load documents from file with proper error handling
            docs = self.load_documents_from_file(self.config.file_path)
            
            # Filter out duplicates (though there shouldn't be any for a new collection)
            unique_docs = self.filter_duplicates(docs)
            
            if not unique_docs:
                logger.warning("No documents to add, data may be empty or all duplicates")
                # Create empty collection
                self.vector_db = QdrantVectorStore.from_documents(
                    [Document(page_content="Initial document")],
                    self.embeddings,
                    url=self.config.url,
                    prefer_grpc=self.config.prefer_grpc,
                    collection_name=self.config.collection_name
                )
                # Remove the initial document
                if hasattr(self.vector_db, "_client"):
                    points = self.client.scroll(
                        collection_name=self.config.collection_name,
                        limit=1
                    )[0]
                    if points:
                        self.client.delete(
                            collection_name=self.config.collection_name,
                            points_selector=points[0].id
                        )
            else:
                # Create vector store with unique documents
                self.vector_db = QdrantVectorStore.from_documents(
                    unique_docs,
                    self.embeddings,
                    url=self.config.url,
                    prefer_grpc=self.config.prefer_grpc,
                    collection_name=self.config.collection_name
                )
                
                logger.info(f"Initialized vector DB with {len(unique_docs)} documents")
        
        logger.success(f"Successfully connected to vector store at {self.config.url}/{self.config.collection_name}")
        return self.vector_db
    
    def add_documents(self, file_path: Optional[str] = None) -> None:
        """Add new documents to the vector database, skipping duplicates"""
        if not self.vector_db:
            raise ValueError("Vector DB must be initialized first with initialize_or_get_db()")
        
        # Use provided file path or default from config
        file_path = file_path or self.config.file_path
        
        # Load documents from file with proper error handling
        docs = self.load_documents_from_file(file_path)
        
        # Filter out duplicates
        unique_docs = self.filter_duplicates(docs)
        
        if not unique_docs:
            logger.info("No new documents to add, all are duplicates")
            return
        
        # Add documents to vector store
        self.vector_db.add_documents(unique_docs)
        
        logger.info(f"Added {len(unique_docs)} new documents to vector store")


def main(override_collection_name: Optional[str] = None):
    """Main function to run the vector DB manager"""
    # Load config
    config = Config()
    
    # Override collection name if provided
    if override_collection_name:
        config.collection_name = override_collection_name
    
    # Print debug info
    print(f"Working directory: {os.getcwd()}")
    print(f"Environment COLLECTION_NAME: {os.environ.get('COLLECTION_NAME', 'Not set')}")
    print(f"Config collection_name: {config.collection_name}")
    
    # Initialize manager
    manager = VectorDBManager(config)
    
    # Initialize or get vector DB
    vector_db = manager.initialize_or_get_db()
    
    return vector_db, manager


if __name__ == "__main__":
    vector_db, manager = main()