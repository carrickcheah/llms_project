########################################################################################
# task.py                                                                              #
# Contents of table (Tasks)                                                            #
# 1. generate_sql_query: Generates SQL query from user question (Task).                #
# 2. execute_sql_query: Executes SQL query and returns result (Task).                 #
# 3. process_database_question: Processes database question with SQL (Task).          #
########################################################################################

from langgraph.func import task
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

########################################################################################
##     1. generate_sql_query: Generates SQL query from user question (Task)           ##
########################################################################################

@task
def generate_sql_query(question: str, sql_generation_chain) -> str:
    """Generates an SQL query from the user's question."""
    logger.info(f"Generating SQL for: {question}")
    return sql_generation_chain.invoke({"question": question})

########################################################################################
##     2. execute_sql_query: Executes SQL query and returns result (Task)            ##
########################################################################################

@task
def execute_sql_query(sql_query: str, db) -> str:
    """Executes the SQL query and returns the result."""
    logger.info(f"Executing SQL: {sql_query}")
    try:
        result = db.run(sql_query)
        return result
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}")
        return f"Error executing query: {str(e)}"

########################################################################################
##     3. process_database_question: Processes database question with SQL (Task)     ##
########################################################################################

@task
def process_database_question(question: str, sql_generation_chain, db, llm) -> str:
    """Processes a database question by generating and executing an SQL query, then formatting the response."""
    try:
        sql_query = generate_sql_query(question, sql_generation_chain).result()
        result = execute_sql_query(sql_query, db).result()
        dynamic_prompt = ChatPromptTemplate.from_template("""
        Given an input question and its executed SQL result, return the answer with column names explored from the query.
        Use the following format:

        Question: "{question}"
        SQLQuery: "{sql_query}"
        SQLResult: "{result}"
        Answer: "Final answer incorporating column names"
        Insight: "Optimize the Answer into a simple report, approximately 20 words"
        """)
        response_chain = dynamic_prompt | llm | StrOutputParser()
        response = response_chain.invoke({"question": question, "sql_query": sql_query, "result": result})
        return response
    except Exception as e:
        logger.error(f"Error processing database question: {str(e)}")
        return f"Error: {str(e)}"