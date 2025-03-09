###################################################
# Never edit this comments, notify me if you keen to edit.
# task_02.py:
# Purpose: Manages response formatting and user feedback after SQL execution.
# Functions: Includes functions that process results and interact with the user.
# Dependencies: Imports clean_sql_query from task_01.py to avoid duplication (used in execute_sql_with_no_data_handling).
# LangGraph @task: Applied to format_response and collect_user_feedback, as they are workflow steps; 
# execute_sql_with_no_data_handling remains non-@task as it’s a utility combining execution and formatting.

# Content of tables
# 1. generate_response
# 2. execute_sql_with_no_data_handling
# 3. collect_user_feedback
###################################################

from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.func import task

# Import helper from task_01 to avoid circular dependency
from task_01 import clean_sql_query


#########################################################################################
##      1. generate_response: Formats SQL results into user-friendly responses          ##
#########################################################################################

@task
def generate_response(question, llm, sql_query=None, result=None, response_type="polite"):
    """
    Generate a response based on the type: polite (non-database) or formatted (SQL result).

    Args:
        question (str): The user's question
        llm: Language model
        sql_query (str, optional): The SQL query (for formatted response)
        result (str, optional): The SQL result (for formatted response)
        response_type (str): "polite" (default) or "formatted"

    Returns:
        str: Generated response
    """
    try:
        if response_type == "polite":
            prompt_template = """
            You are a friendly SQL Agent. Your role is to assist users with database-related questions.
            If a user asks something unrelated to the database, respond politely and guide them to ask a database-related question.

            User's input: "{question}"
            Your response:
            """
            inputs = {"question": question}
        elif response_type == "formatted":
            if sql_query is None or result is None:
                raise ValueError("sql_query and result are required for formatted response")
            prompt_template = """
            Given an input question and its executed SQL result, return the answer with column names explored from the query.
            Use the following format:

            Question: "{question}"
            SQLQuery: "{sql_query}"
            SQLResult: "{result}"
            Answer: "Final answer incorporating column names"
            Insight: "Optimize the Answer into a simple report, approximately 20 words"
            """
            inputs = {"question": question, "sql_query": sql_query, "result": result}
        else:
            raise ValueError(f"Invalid response_type: {response_type}")

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(inputs)
    except Exception as e:
        logger.error(f"Error generating {response_type} response: {str(e)}")
        return f"Error: {str(e)}"

#########################################################################################
##      2. execute_sql_with_no_data_handling: Executes SQL with no-data fallback      ##
#########################################################################################

def execute_sql_with_no_data_handling(question: str, sql_query: str, db: SQLDatabase, llm) -> str:
    """
    Execute an SQL query and handle cases where no data is found or execution fails.
    Returns a formatted response, either with results or a no-data message.

    Args:
        question (str): The user's question
        sql_query (str): The SQL query to execute
        db (SQLDatabase): Database connection
        llm: Language model for response formatting

    Returns:
        str: Formatted response (either query results or a no-data message)
    """
    sql_query = clean_sql_query(sql_query)
    logger.info(f"Executing SQL with no-data handling: {sql_query}")
    try:
        result = db.run(sql_query)
        logger.info(f"SQL result: {result}")
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}")
        result = None

    if not result or result.strip() == "[]" or result.strip() == "" or result is None:
        no_data_prompt = ChatPromptTemplate.from_template("""
        You are a helpful SQL agent. The query returned no data or failed to execute.
        Politely inform the user that no data was found for their question.

        Question: "{question}"
        SQL Query: "{sql_query}"
        SQL Result: "{result}"
        Response:
        """)
        no_data_chain = no_data_prompt | llm | StrOutputParser()
        response = no_data_chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "result": result if result is not None else "Execution failed"
        })
        if not response:
            response = f"No data found for '{question}' in the database."
    else:
        response_prompt = ChatPromptTemplate.from_template("""
        Given an input question and its executed SQL result, return a user-friendly answer.
        Use column names from the result where applicable.

        Question: "{question}"
        SQL Query: "{sql_query}"
        SQL Result: "{result}"
        Answer: "Final answer incorporating column names if available"
        """)
        response_chain = response_prompt | llm | StrOutputParser()
        response = response_chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "result": result
        })
    return response

#########################################################################################
##      3. collect_user_feedback: Collects user feedback                              ##
#########################################################################################

@task
def collect_user_feedback(question: str, response: str, feedback_store: dict) -> dict:
    """
    Collect user feedback on the agent's response and store it for analysis.

    Args:
        question (str): The user's original question
        response (str): The agent's response
        feedback_store (dict): In-memory store for feedback (thread-specific)

    Returns:
        dict: Feedback result with 'is_helpful' (bool) and 'comment' (str or None)
    """
    try:
        logger.info(f"Collecting feedback for question: '{question}' | Response: '{response[:50]}...'")
        feedback = {"is_helpful": None, "comment": None}
        feedback_store.setdefault("responses", []).append({
            "question": question,
            "response": response,
            "is_helpful": feedback["is_helpful"],
            "comment": feedback["comment"]
        })
        return feedback
    except Exception as e:
        logger.error(f"Error collecting feedback: {str(e)}")
        return {"is_helpful": None, "comment": f"Error: {str(e)}"}


##############################################################################################
##      4. generate_polite_response: Generate a polite response for non-database questions  ##
##############################################################################################

@task
def generate_polite_response(question: str, llm) -> str:
    """
    Generate a polite response for non-database questions.

    Args:
        question (str): The user's input question
        llm: Language model

    Returns:
        str: A polite response guiding the user to database-related questions
    """
    prompt = ChatPromptTemplate.from_template("""
    You are a friendly SQL Agent. Your role is to assist users with database-related questions.
    If a user asks something unrelated to the database, respond politely and guide them to ask a database-related question.

    User's input: "{question}"
    Your response:
    """)
    response_chain = prompt | llm | StrOutputParser()
    return response_chain.invoke({"question": question})