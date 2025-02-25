import os
from typing import Annotated, List, Literal
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import RateLimitError
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

# LangChain / LangGraph Imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ------------------------------
# Configuration
# ------------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hotela"

DATABASE_URI = "mysql+pymysql://toolbox_user:my-password@127.0.0.1:3306/toolbox_db"

# ------------------------------
# Database & Qdrant Setup
# ------------------------------
db = SQLDatabase.from_uri(DATABASE_URI)
qdrant_client = QdrantClient(url=QDRANT_URL)

# ------------------------------
# LLM Setup
# ------------------------------
model = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_retries=3,
    request_timeout=30,
    max_tokens=1024,
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_model_invoke(prompt: str) -> str:
    response = model.invoke(prompt)
    return response.content

# ------------------------------
# Tools
# ------------------------------
class QueryResult(BaseModel):
    result: str = Field(..., description="The result of the SQL query")

@tool(args_schema=QueryResult)
def db_query_tool(query: str) -> str:
    """Execute a SQL query against the sales database and return results"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Query Error: {str(e)}"

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")

# ------------------------------
# Graph State Definition
# ------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[AIMessage], add_messages]

# ------------------------------
# Fallback / Error Handling
# ------------------------------
def handle_tool_error(state) -> dict:
    """Fallback if a tool node fails."""
    error = state.get("error")
    last_message = state["messages"][-1]

    # If the last message had tool_calls, respond with a ToolMessage referencing them
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
        return {
            "messages": [
                ToolMessage(
                    content=f"Tool Error: {repr(error)}",
                    tool_call_id=tc["id"],
                )
                for tc in last_message.tool_calls
            ]
        }
    # Otherwise, produce a generic fallback ToolMessage
    return {
        "messages": [
            ToolMessage(
                content=f"Tool Error: {repr(error)}",
                tool_call_id="fallback_tool_call",
            )
        ]
    }

def create_tool_node_with_fallback(tool_list: list) -> RunnableWithFallbacks:
    return ToolNode(tool_list).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
        exception_key="error"
    )

# ------------------------------
# Helper Node: Debug Messages
# ------------------------------
import json
def debug_messages(state: AgentState) -> dict:
    """Utility node to print out the current message list for debugging."""
    print("📝 Current Messages:")
    print(json.dumps([m.dict() for m in state["messages"]], indent=2))
    return state

# ------------------------------
# Analyze Schema Node
# ------------------------------
def model_get_schema_node(state: AgentState) -> dict:
    try:
        schema = db.get_table_info()
        # If the schema is too large, truncate
        if len(schema) > 4000:
            schema = schema[:2000] + "\n...[TRUNCATED]..." + schema[-2000:]
            
        prompt = f"""Analyze this database schema and identify relevant tables:
        Schema: {schema}
        Question: {state['messages'][-1].content}
        Relevant Tables:"""
        
        response = safe_model_invoke(prompt)
        return {"messages": [AIMessage(content=response)]}
    except RateLimitError:
        return {"messages": [AIMessage(content="Error: Rate limit exceeded")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"System Error: {str(e)}")]}

# ------------------------------
# Query Generation
# ------------------------------
query_gen_system = """You are a SQL expert. Generate SQL queries based on the user's question.

Rules:
1. Use MySQL syntax.
2. Never modify data (READONLY).
3. Limit results to 5 rows.
4. Handle NULLs properly.
5. Use SQL parameters if needed."""

query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", query_gen_system),
    ("placeholder", "{messages}")
])

# We want the chain to produce an AIMessage with a tool call to "db_query_tool".
class FinalAnswer(BaseModel):
    answer: str = Field(..., description="The final answer to present to the user")

query_gen_chain = query_gen_prompt | model.bind_tools([FinalAnswer])

# ------------------------------
# Graph Construction
# ------------------------------
graph = StateGraph(AgentState)

# 1) start
graph.add_node("start", lambda _: {
    "messages": [AIMessage(
        content="Initializing database connection...",
        tool_calls=[{"name": "sql_db_list_tables", "args": {}, "id": "init"}]
    )]
})

# 2) list_tables
graph.add_node("list_tables", create_tool_node_with_fallback([list_tables_tool]))

# 3) get_schema
graph.add_node("get_schema", create_tool_node_with_fallback([get_schema_tool]))

# 4) analyze_schema
graph.add_node("analyze_schema", model_get_schema_node)

# 5) debug_messages (optional debugging)
graph.add_node("debug_messages", debug_messages)

# 6) generate_query
graph.add_node("generate_query", query_gen_chain)

# 7) execute_query
graph.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

# Edges
graph.add_edge(START, "start")
graph.add_edge("start", "list_tables")
graph.add_edge("list_tables", "get_schema")
graph.add_edge("get_schema", "analyze_schema")

# Go through debug_messages before generate_query
graph.add_edge("analyze_schema", "debug_messages")
graph.add_edge("debug_messages", "generate_query")

# ------------------------------
# Routing Logic
# ------------------------------
def route_state(state: AgentState) -> Literal["generate_query", "execute_query", "END"]:
    """
    This function decides what node to go to next based on the last message.
    """
    last_message = state["messages"][-1]
    
    # If it's a tool error, re-attempt query generation
    if isinstance(last_message, ToolMessage) and "error" in last_message.content.lower():
        return "generate_query"
    
    # If the last AIMessage has a tool call, it means we're ready to execute
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
        return "execute_query"
    
    # Otherwise, keep refining the query or exit if we can't proceed
    return "generate_query"

graph.add_conditional_edges(
    "generate_query",
    route_state,
    {
        "generate_query": "generate_query",
        "execute_query": "execute_query",
        "END": END
    }
)

graph.add_edge("execute_query", END)

# Compile the graph into an app
app = graph.compile()

# ------------------------------
# Main Entry Point
# ------------------------------
if __name__ == "__main__":
    from langchain_community.callbacks import get_openai_callback
    
    question = input("Enter your question: ")
    
    try:
        with get_openai_callback() as cb:
            response = app.invoke({"messages": [("user", question)]})
            # The final answer is usually stored in the last message's tool_calls
            final_ai_msg = response["messages"][-1]
            
            if hasattr(final_ai_msg, "tool_calls"):
                final_answer = final_ai_msg.tool_calls[0]["args"]["answer"]
                print("Final Answer:", final_answer)
            else:
                print("No final answer found in tool calls. Full response:\n", response)
            
            print(f"\nToken Usage:\n{cb}")
                
    except Exception as e:
        print(f"Application Error: {str(e)}")
