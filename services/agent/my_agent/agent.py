import os
from typing import Annotated, List, Literal
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import RateLimitError
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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

# Other imports
from loguru import logger

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URI = "mysql+pymysql://myuser:mypassword@127.0.0.1:3306/nex_valiant"
db = SQLDatabase.from_uri(DATABASE_URI)
# logger.success(db.dialect)
# logger.info(db.get_usable_table_names())

# Configure OpenAI client
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_retries=3,
    request_timeout=30,
    max_tokens=1024,
)

################################################################################################################
################################################################################################################
# Utility functions
# We will define a few utility functions to help us with the agent implementation. 
# Specifically, we will wrap a ToolNode with a fallback to handle errors and surface them to the agent.

from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

################################################################################################################
################################################################################################################
"""
Define tools for the agent
We will define a few tools that the agent will use to interact with the database.

1)  list_tables_tool: Fetch the available tables from the database
2)  get_schema_tool: Fetch the DDL for a table
3)  db_query_tool: Execute the query and fetch the results OR return an error message if the query fails

For the first two tools, we will grab them from the SQLDatabaseToolkit, also available in the langchain_community package.
"""

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

# logger.info(list_tables_tool.invoke(""))
# logger.info(get_schema_tool.invoke("final_bom"))



################################################################################################################
################################################################################################################
# The third will be defined manually. 
# For the db_query_tool, we will execute the query against the database and return the results.

from langchain_core.tools import tool


@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result


# logger.success(db_query_tool.invoke("SELECT * FROM final_bom LIMIT 10;"))




################################################################################################################
################################################################################################################

# While not strictly a tool, we will prompt an LLM to check for common mistakes in the query and later add this as a node in the workflow.

from langchain_core.prompts import ChatPromptTemplate

query_check_system = """
You are a meticulous SQL expert. When reviewing a MySQL query, ensure you check for:
- Misuse of NOT IN with NULL values
- Using UNION instead of UNION ALL when appropriate
- Incorrect use of BETWEEN for exclusive ranges
- Data type mismatches in predicates
- Improperly quoted identifiers
- Incorrect argument count in functions
- Faulty type casting
- Joining on the wrong columns
- Ambiguous column references in multi-table queries
- Missing or incorrect join conditions that may cause Cartesian products
- Errors in aggregate function usage and GROUP BY clauses
- Potential performance pitfalls (e.g., missing indexes)
- SQL injection vulnerabilities when handling dynamic input

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [db_query_tool], tool_choice="required"
)

query_check.invoke({"messages": [("user", "SELECT * FROM final_bom LIMIT 10;")]})





































