from loguru import logger
import time
from qdrant_client import QdrantClient
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore


def qdrant_on_prem(
    embedding: Embeddings,
    collection_name: str,
    url: str = "http://localhost:6333",
    prefer_grpc: bool = True,
    timeout: float = 10.0,
    retry_attempts: int = 3
) -> QdrantVectorStore:
    """
    Retrieve or initialize an existing Qdrant vector store.
    
    Args:
        embedding: Embeddings model for vector encoding
        url: URL of the Qdrant server
        collection_name: Name of the collection to connect to
        prefer_grpc: Whether to use gRPC protocol instead of HTTP
        timeout: Connection timeout in seconds
        retry_attempts: Number of connection retry attempts
        
    Returns:
        QdrantVectorStore: Connected vector store instance
        
    Raises:
        ValueError: If embedding is not provided
        ConnectionError: If connection to Qdrant fails
    """
    if not embedding:
        logger.error("Embedding model is not provided.")
        raise ValueError("An embedding model must be provided.")

    # Validate collection name
    if not collection_name:
        logger.error("Collection name is empty or not provided.")
        raise ValueError("A valid collection name must be provided.")
    
    attempt = 0
    last_exception = None
    
    while attempt < retry_attempts:
        try:
            logger.info(f"Connecting to Qdrant vector store at {url} with collection '{collection_name}' "
                       f"(prefer_grpc={prefer_grpc}, attempt={attempt+1}/{retry_attempts})")
            
            # Initialize client with timeout
            client = QdrantClient(
                url=url, 
                prefer_grpc=prefer_grpc,
                timeout=timeout
            )
            
            # Verify connection by making a simple API call
            client.get_collections()
            
            # Create vector store instance
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding
            )
            
            # Optional: Check if collection exists
            collections = client.get_collections().collections
            collection_exists = any(collection.name == collection_name for collection in collections)
            
            if not collection_exists:
                logger.warning(f"Collection '{collection_name}' does not exist in Qdrant. "
                             f"It will be created when documents are added.")
            
            logger.success(f"Successfully connected to Qdrant vector store '{collection_name}'.")
            return qdrant_store
        
        except Exception as e:
            attempt += 1
            last_exception = e
            logger.warning(f"Connection attempt {attempt}/{retry_attempts} failed: {str(e)}")
            
            if attempt < retry_attempts:
                # Wait before retrying (exponential backoff)
                wait_time = 0.5 * (2 ** attempt)  # 1s, 2s, 4s, ...
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to connect to Qdrant vector store after {retry_attempts} attempts: {str(e)}")
                raise ConnectionError(f"Failed to connect to Qdrant vector store: {str(e)}") from last_exception











