import asyncio
from typing import Annotated
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# LangChain + Tools
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Core message and tool types
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

# Qdrant configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hotela"

##############################################################################
# 1) MySQL connection
##############################################################################
# Database configuration
DATABASE_URI = "mysql+pymysql://toolbox_user:my-password@127.0.0.1:3306/toolbox_db"
db = SQLDatabase.from_uri(DATABASE_URI)

##############################################################################
# 2) LLM and SQL Toolkit
##############################################################################
llm = ChatOllama(model="llama3.1", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

##############################################################################
# 3) Qdrant Client Initialization
##############################################################################
qdrant_client = QdrantClient(url=QDRANT_URL)

# Create Qdrant collection if it doesn't exist
try:
    qdrant_client.get_collection(COLLECTION_NAME)
except Exception:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

##############################################################################
# 4) Custom Qdrant Tools
##############################################################################
@tool
def search_hotels_by_vector(query: str) -> str:
    """Search hotels using vector similarity in Qdrant."""
    try:
        # Generate embeddings for the query (replace with your embedding model)
        # For simplicity, we assume the query is already an embedding
        query_embedding = [0.0] * 768  # Replace with actual embedding generation logic

        # Perform vector search in Qdrant
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=5,
        )

        # Format and return results
        results = []
        for hit in search_result:
            results.append(f"Hotel ID: {hit.id}, Score: {hit.score}")
        return "\n".join(results) if results else "No hotels found."
    except Exception as e:
        return f"Error: {str(e)}"

# Add Qdrant tools
tools.append(search_hotels_by_vector)

##############################################################################
# 5) Graph State
##############################################################################
class MyState(TypedDict):
    messages: Annotated[list, add_messages]

##############################################################################
# 6) Agent Node
##############################################################################
def agent_node(state: MyState):
    """ReAct agent node with SQL and Qdrant tools."""
    agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
    result = agent.invoke({"messages": state["messages"]}, stream_mode="values")
    return {"messages": [result["messages"][-1]]}

##############################################################################
# 7) Build Graph
##############################################################################
graph_builder = StateGraph(MyState)
graph_builder.add_node("agent_node", agent_node)
graph_builder.add_edge(START, "agent_node")
graph_builder.add_edge("agent_node", END)
graph = graph_builder.compile(checkpointer=MemorySaver())

##############################################################################
# 8) Main Interaction Loop
##############################################################################
def main():
    config = {"configurable": {"thread_id": "mysql-thread"}}
    print("MySQL + Qdrant Agent: Ask a question about your database or hotels. (Press Ctrl+C to exit)\n")
    conversation = []
    
    while True:
        # Remove brackets from user input
        user_query = input("User: ").strip().replace('[', '').replace(']', '')
        if not user_query:
            continue
            
        conversation.append(("user", user_query))
        
        events = graph.stream({"messages": conversation}, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                last_msg = event["messages"][-1]
                if last_msg.role == "assistant":
                    print(f"Assistant: {last_msg.content}")
                    conversation.append((last_msg.role, last_msg.content))

if __name__ == "__main__":
    main()