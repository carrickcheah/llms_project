import json
from uuid import uuid4
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client.http.models import Distance, VectorParams
from loguru import logger

# Load environment variables
load_dotenv()

# Set up the OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Set up the Qdrant client
url = "http://localhost:6333"

# Load the JSON file
json_file_path = "/Users/carrickcheah/llms_project/services/agent/qdran/abdd.json"
with open(json_file_path, "r") as file:
    data = json.load(file)

# Since data is a list of tool definitions, use it directly
tools = data

# Convert tools into LangChain Document objects
docs = []
for tool in tools:
    # Create a meaningful text representation of the tool
    page_content = (
        f"Operation: {tool['operation']}\n"
        f"User Query: {tool['user_query']}\n"
        f"Generated SQL: {tool['generated_sql']}\n"
        f"Query Type: {tool['query_type']}\n"
        f"Explanation: {tool['explanation']}"
    )
    # Include metadata for filtering or additional context
    metadata = {
        "operation": tool["operation"],
        "query_type": tool["query_type"],
        "tables_used": tool["tables_used"],
        "columns_used": tool["columns_used"],
        "difficulty_level": tool["difficulty_level"],
        "schema_name": tool["schema_name"],
        "execute_in_mysql": tool["execute_in_mysql"],
    }
    docs.append(Document(page_content=page_content, metadata=metadata))

# Add documents to the Qdrant vector store
qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    collection_name="hotela",
    force_recreate=True
)

logger.success("Documents ingested successfully!")