from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import json
import logging
from pathlib import Path

# Initialize environment
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class QdrantSettings(BaseSettings):
    """Qdrant configuration settings"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="QDRANT_"
    )

    url: str = "http://localhost:6333"
    collection_name: str = "query_embeddings"
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    content_payload_key: str = "page_content"
    hf_model_name: str = "carrick113/autotrain-wsucv-dqrgc"
    file_path: str = "/Users/carrickcheah/llms_project/services/agent/vector/abc.json"

def get_embeddings(settings: QdrantSettings) -> Embeddings:
    """Initialize Hugging Face embeddings"""
    return HuggingFaceEmbeddings(model_name=settings.hf_model_name)

def get_qdrant_client(settings: QdrantSettings) -> QdrantClient:
    """Create and return Qdrant client"""
    return QdrantClient(
        url=settings.url,
        prefer_grpc=settings.prefer_grpc,
        api_key=settings.api_key
    )

def load_documents(file_path: str) -> List[Document]:
    """Load and validate SQL query documents from JSON file"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    if not isinstance(data, list):
        raise ValueError("JSON root should be a list of query objects")

    documents = []
    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not a dictionary")

        # Validate required fields
        required_fields = ["user_query", "generated_sql", "tables_used"]
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Item {idx} missing required field: {field}")

        # Create document structure
        metadata = {
            "generated_sql": item["generated_sql"],
            "tables_used": ", ".join(item["tables_used"]),
            "columns_used": ", ".join(item.get("columns_used", [])),
            "schema_name": item.get("schema_name", "unknown"),
            "execute_in_mysql": str(item.get("execute_in_mysql", False))
        }

        documents.append(Document(
            page_content=item["user_query"],
            metadata=metadata
        ))

    logger.info(f"Loaded {len(documents)} valid SQL query documents")
    return documents

def connect_vector_store(
    settings: QdrantSettings,
    embeddings: Embeddings
) -> QdrantVectorStore:
    """Connect to or initialize Qdrant vector store"""
    client = get_qdrant_client(settings)
    
    try:
        # Check for existing collection
        collections = client.get_collections().collections
        if any(c.name == settings.collection_name for c in collections):
            logger.info(f"Using existing collection: {settings.collection_name}")
            return QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=settings.collection_name,
                url=settings.url,
                api_key=settings.api_key,
                content_payload_key=settings.content_payload_key
            )
    except Exception as e:
        logger.warning(f"Collection check failed: {e}")

    # Create new collection with documents
    logger.info(f"Initializing new collection: {settings.collection_name}")
    documents = load_documents(settings.file_path)
    
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=settings.url,
        collection_name=settings.collection_name,
        api_key=settings.api_key,
        content_payload_key=settings.content_payload_key
    )

def main():
    """Main application entry point"""
    try:
        settings = QdrantSettings()
        embeddings = get_embeddings(settings)
        vector_store = connect_vector_store(settings, embeddings)
        logger.info("Vector store initialization completed successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()