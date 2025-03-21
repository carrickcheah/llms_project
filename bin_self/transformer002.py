import os
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import RateLimitError
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# LangChain imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URI = "mysql+pymysql://myuser:mypassword@127.0.0.1:3306/nex_valiant"  # Updated to nex_valiant
db = SQLDatabase.from_uri(DATABASE_URI)

# Configure OpenAI client
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_retries=3,
    request_timeout=30,
    max_tokens=1024,
)

# Retry configuration for OpenAI calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_model_invoke(prompt: str) -> str:
    response = model.invoke(prompt)
    return response.content

# Define a Pydantic model for query results
class QueryResult(BaseModel):
    query: str = Field(description="The SQL query to execute")

@tool(args_schema=QueryResult)
def db_query_tool(query: str) -> str:
    """Execute a SQL query against the database and return results"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Query Error: {str(e)}"

# Initialize SQL toolkit and tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="carrick113/autotrain-wsucv-dqrgc")

# Path to your JSON file
file_path = "/Users/carrickcheah/llms_project/services/agent/vector/abc.json"
collection_name = "query_embeddings"

# Generate a unique ID for each document
def generate_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# Load JSON data into Qdrant with deduplication
def load_json_to_qdrant():
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        text_content=False,
        metadata_func=lambda record, additional_fields: {
            "id": generate_unique_id(record.get("user_query", "")),
            "user_query": record.get("user_query", ""),
            "generated_sql": record.get("generated_sql", ""),
            "tables_used": record.get("tables_used", []),
            "columns_used": record.get("columns_used", []),
            "schema_name": record.get("schema_name", ""),
            "execute_in_mysql": record.get("execute_in_mysql", False),
        }
    )
    
    docs = loader.load()
    client = QdrantClient(url="http://localhost:6333")
    
    points = [
        PointStruct(
            id=doc.metadata["id"],
            vector=embeddings.embed_query(doc.page_content),
            payload=doc.metadata
        )
        for doc in docs
    ]
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print("Documents upserted successfully into Qdrant.")
    
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url="http://localhost:6333",
        prefer_grpc=True
    )

# Process user query
def process_query(user_query: str, qdrant: QdrantVectorStore) -> str:
    # Search Qdrant for similar queries
    results = qdrant.similarity_search(user_query, k=3)
    
    # Filter results where execute_in_mysql is True
    filtered_results = [
        result for result in results
        if result.metadata.get("execute_in_mysql", False) is True
    ]
    
    if filtered_results:
        # Use the most relevant result
        top_result = filtered_results[0]
        sql_query = top_result.metadata["generated_sql"]
        result = db_query_tool.invoke({"query": sql_query})
        return f"Executed SQL query: {sql_query}\nResult: {result}"
    else:
        # Fallback to OpenAI to generate a response
        prompt = f"User query: {user_query}\nGenerate a SQL query or answer based on the schema: {db.get_table_info()}"
        return safe_model_invoke(prompt)

# Main interaction loop
def main():
    print("Starting interaction with the database. Type 'exit' to stop.")
    
    # Load Qdrant once at startup
    qdrant = load_json_to_qdrant()
    
    while True:
        user_input = input("User: ").strip().replace('[', '').replace(']', '')
        if not user_input or user_input.lower() == "exit":
            break
        
        # Process the user query
        response = process_query(user_input, qdrant)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()