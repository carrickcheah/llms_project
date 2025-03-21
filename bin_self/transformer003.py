# streamllit chat with embedded SQL execution

import os
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import RateLimitError
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import streamlit as st
from datetime import datetime
import time

# LangChain imports (non-PyTorch related)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import JSONLoader
from langchain_qdrant import QdrantVectorStore

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URI = "mysql+pymysql://myuser:mypassword@127.0.0.1:3306/nex_valiant"
db = SQLDatabase.from_uri(DATABASE_URI)

# Configure OpenAI client
model = ChatOpenAI(
    model="gpt-4",
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

# Path to your JSON file
file_path = "/Users/carrickcheah/llms_project/services/agent/vector/abc.json"
collection_name = "query_embeddings"

# Generate a unique ID for each document
def generate_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# Load JSON data into Qdrant with deduplication (delayed PyTorch import)
def load_json_to_qdrant():
    from langchain_huggingface import HuggingFaceEmbeddings  # Import here to delay PyTorch loading
    embeddings = HuggingFaceEmbeddings(model_name="carrick113/autotrain-wsucv-dqrgc")
    
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
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url="http://localhost:6333",
        prefer_grpc=True
    )

# Process user query
def process_query(user_query: str, qdrant: QdrantVectorStore) -> str:
    results = qdrant.similarity_search(user_query, k=3)
    
    filtered_results = [
        result for result in results
        if result.metadata.get("execute_in_mysql", False) is True
    ]
    
    if filtered_results:
        top_result = filtered_results[0]
        sql_query = top_result.metadata["generated_sql"]
        result = db_query_tool.invoke({"query": sql_query})
        return f"Executed SQL query: {sql_query}\nResult: {result}"
    else:
        prompt = f"User query: {user_query}\nGenerate a SQL query or answer based on the schema: {db.get_table_info()}"
        return safe_model_invoke(prompt)

# Display message in Streamlit
def display_message(role: str, content: str):
    if role == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
    elif role == "thought":
        with st.chat_message("assistant"):
            st.markdown(f'<div class="thought-message">🤔 {content}</div>', unsafe_allow_html=True)

# Generate thought process
def generate_thought_process(prompt: str) -> List[str]:
    return [
        "Analyzing the user query...",
        "Checking Qdrant for similar queries...",
        "Determining if SQL execution is required...",
    ]

# Main Streamlit function
def main():
    st.set_page_config(page_title="Nex AI Agent Chat")
    
    # Database connection
    try:
        db = SQLDatabase.from_uri(DATABASE_URI)
        st.write("Connected to the database! Yay!")
    except Exception as e:
        st.write(f"Oh no! Couldn’t connect: {str(e)}")
        return
    
    # Qdrant setup
    try:
        vector_store = load_json_to_qdrant()
        st.write("Qdrant vector store initialized!")
    except Exception as e:
        st.write(f"Qdrant setup failed: {str(e)}")
        return
    
    # Custom CSS for chat styling
    st.markdown("""
        <style>
        .stChatMessage {
            max-width: 80%;
            margin: 10px;
        }
        .user-message {
            background-color: #2b313e;
            border-radius: 10px;
            padding: 10px;
            animation: fadeIn 0.5s ease-in-out;
        }
        .assistant-message {
            background-color: #1a1e26;
            border-radius: 10px;
            padding: 10px;
            animation: fadeIn 0.5s ease-in-out;
        }
        .thought-message {
            background-color: #3a3f4a;
            border-radius: 10px;
            padding: 10px;
            color: #ffffff;
            font-style: italic;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Nex AI Agent")
    st.caption(f"Powered by GPT-4 with Qdrant RAG and DB | Date: {datetime.now().strftime('%B %d, %Y')}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI assistant with database access. How can I assist you today?"}
        ]
    
    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])
    
    # Handle user input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)
        
        # Show thought process
        thought_placeholder = st.empty()
        thought_steps = generate_thought_process(prompt)
        
        for step in thought_steps:
            with thought_placeholder.container():
                display_message("thought", step)
            time.sleep(0.8)
        
        # Process the query and display response
        with st.spinner("Finalizing response..."):
            if "how many" in prompt.lower() and "table" in prompt.lower():
                tables = db.get_usable_table_names()
                response = f"There are {len(tables)} tables in the database. Here they are:\n\n{', '.join(tables)}"
            else:
                response = process_query(prompt, vector_store)
            
            full_response = (
                f"**Thought Process:**\n\n" + "\n".join(thought_steps) +
                f"\n\n**Answer:**\n\n{response}"
            )
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            display_message("assistant", full_response)

if __name__ == "__main__":
    main()