import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Path to your JSON file
file_path = "/Users/carrickcheah/llms_project/services/agent/vector/abc.json"

# Define the loader with corrected metadata_func
loader = JSONLoader(
    file_path=file_path,
    jq_schema=".[]",  # Load all items in the array
    text_content=False,
    metadata_func=lambda record, additional_fields: {
        "user_query": record.get("user_query"),
        "generated_sql": record.get("generated_sql"),
        "tables_used": record.get("tables_used", []),
        "columns_used": record.get("columns_used", []),
        "schema_name": record.get("schema_name"),
        "execute_in_mysql": record.get("execute_in_mysql", False),
    }
)

# Load documents
docs = loader.load()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="carrick113/autotrain-u1whw-a7kfd")

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")
collection_name = "query_embeddings"

# Generate unique IDs and upsert documents
points = [
    PointStruct(
        id=hashlib.md5(doc.metadata["user_query"].encode()).hexdigest(),  # Unique ID
        vector=embeddings.embed_query(doc.page_content),
        payload=doc.metadata
    )
    for doc in docs
]

client.upsert(
    collection_name=collection_name,
    points=points
)
print("Documents upserted successfully.")