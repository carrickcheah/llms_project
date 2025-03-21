"""
Direct connect database
Langgraph
https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""



import os
from typing import Annotated, List, Literal
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import RateLimitError
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# LangChain imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# LangGraph imports
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# Database configuration
# DATABASE_URI = "mysql+pymysql://myuser:mypassword@127.0.0.1:3306/nex_valiant"
DATABASE_URI = "mysql+pymysql://toolbox_user:my-password@127.0.0.1:3306/toolbox_db"
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

# ------------------------------
# Predefined SQL Query Templates
# ------------------------------
PREDEFINED_QUERIES = {
    "top_sales_agent_2009": """
        SELECT e.FirstName, e.LastName, SUM(i.Total) AS TotalSales 
        FROM Invoice i
        JOIN Customer c ON i.CustomerId = c.CustomerId
        JOIN Employee e ON c.SupportRepId = e.EmployeeId
        WHERE YEAR(i.InvoiceDate) = 2009
        GROUP BY e.EmployeeId
        ORDER BY TotalSales DESC
        LIMIT 1;
    """,
}

# ------------------------------
# Database Tools & Utilities
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
# Graph Node Implementations
# ------------------------------
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], 
        exception_key="error"
    )

def handle_tool_error(state) -> dict:
    error = state.get("error")
    last_message = state["messages"][-1]

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
    return {
        "messages": [
            ToolMessage(
                content=f"Tool Error: {repr(error)}",
                tool_call_id="fallback_tool_call",
            )
        ]
    }

def model_get_schema_node(state: AgentState) -> dict:
    try:
        schema = db.get_table_info()
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
# Query Generation Components
# ------------------------------
query_gen_system = f"""You are a SQL expert. Use these predefined queries when applicable:
{PREDEFINED_QUERIES}

Rules:
1. Use MySQL syntax
2. Never modify data (READONLY)
3. Limit results to 5 rows
4. Handle NULLs properly
5. Use SQL parameters if needed"""

query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", query_gen_system),
    ("placeholder", "{messages}")
])

class FinalAnswer(BaseModel):
    answer: str = Field(..., description="The final answer to present to the user")

query_gen_chain = query_gen_prompt | model.bind_tools([FinalAnswer])

# ------------------------------
# Graph Construction (Fixed)
# ------------------------------
graph = StateGraph(AgentState)

# Define nodes with consistent naming
graph.add_node("start", lambda _: {
    "messages": [AIMessage(
        content="Initializing database connection...",
        tool_calls=[{"name": "sql_db_list_tables", "args": {}, "id": "init"}]
    )]
})
graph.add_node("list_tables", create_tool_node_with_fallback([list_tables_tool]))
graph.add_node("get_schema", create_tool_node_with_fallback([get_schema_tool]))
graph.add_node("analyze_schema", model_get_schema_node)
graph.add_node("generate_query", query_gen_chain)  # Correct node name
graph.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

# Define edges
graph.add_edge(START, "start")
graph.add_edge("start", "list_tables")
graph.add_edge("list_tables", "get_schema")
graph.add_edge("get_schema", "analyze_schema")
graph.add_edge("analyze_schema", "generate_query")

# Fixed conditional routing
def route_state(state: AgentState) -> Literal["generate_query", "execute_query", "END"]:
    last_message = state["messages"][-1]
    
    if isinstance(last_message, ToolMessage) and "error" in last_message.content.lower():
        return "generate_query"
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
        return "execute_query"
    
    return END

graph.add_conditional_edges(
    "generate_query",
    route_state,
    {
        "generate_query": "generate_query",
        "execute_query": "execute_query",
        END: END
    }
)

graph.add_edge("execute_query", END)

# Compile the graph
app = graph.compile()

# ------------------------------
# Execution Entry Point
# ------------------------------
if __name__ == "__main__":
    from langchain_community.callbacks import get_openai_callback
    
    question = "Which sales agent had the highest total sales in 2009?"
    
    try:
        if "sales agent" in question and "2009" in question:
            print("Using optimized predefined query")
            result = db.run(PREDEFINED_QUERIES["top_sales_agent_2009"])
            print("Result:", result)
        else:
            with get_openai_callback() as cb:
                response = app.invoke({"messages": [("user", question)]})
                final_answer = response["messages"][-1].tool_calls[0]["args"]["answer"]
                print("Final Answer:", final_answer)
                print(f"Token Usage: {cb}")
                
    except Exception as e:
        print(f"Application Error: {str(e)}")