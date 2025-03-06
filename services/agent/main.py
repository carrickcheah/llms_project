# main.py ( always keep this # comment )
import os
import re
from loguru import logger
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver

# Custom utility imports
from utils.llm import initialize_openai, initialize_embeddings
from utils.maria import initialize_database
from utils.vector_database import qdrant_on_prem
from utils.tools import load_sql_examples, is_database_question
from utils.task import (
    find_similar_examples,
    extract_relevant_tables,
    execute_sql_query,
    validate_sql_with_llm,  # Import the new validation function
)

# Global initialization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm = initialize_openai(os.getenv("OPENAI_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

def extract_tables_from_sql(sql_query: str) -> list:
    """Extract table names from an SQL query."""
    table_pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s|;|$|\n)'
    tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
    tables = list(set(t.strip() for t in tables if t.strip()))
    logger.info(f"Extracted tables from SQL: {tables}")
    return tables

def search_examples_for_sql(question: str, sql_examples) -> tuple[str, list]:
    """Search sql_examples for an exact match and return its SQL and tables."""
    for example in sql_examples:
        if example["page_content"].strip().lower() == question.strip().lower():
            sql_query = example["sql_query"]
            logger.info(f"Exact match found in examples: '{sql_query}'")
            tables = extract_tables_from_sql(sql_query)
            return sql_query, tables
    logger.warning(f"No exact match found in examples for '{question}'")
    return None, []

def generate_polite_response(question: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
    You are a friendly SQL Agent. Your role is to assist users with database-related questions.
    If a user asks something unrelated to the database, respond politely and guide them to ask a database-related question.

    User's input: "{question}"
    Your response:
    """)
    response_chain = prompt | llm | StrOutputParser()
    return response_chain.invoke({"question": question})

@entrypoint(checkpointer=MemorySaver())
def sql_agent_workflow(inputs: dict) -> dict:
    question = inputs.get("question", "").strip()
    if not question:
        return {"response": "Please provide a question."}

    if is_database_question(question):
        logger.info(f"Database question detected: {question}")
        
        # Step 1: Vector search for similar examples
        similar_examples = find_similar_examples(question, qdrant_store, embeddings)
        
        # Step 2: Check for exact match in vector results
        sql_query, tables = None, []
        if similar_examples:
            for example in similar_examples:
                if example["question"].strip().lower() == question.strip().lower():
                    sql_query = example["sql"]
                    tables = extract_tables_from_sql(sql_query)
                    logger.info(f"Exact match found in Qdrant: '{sql_query}'")
                    break
        
        # Step 3: Fallback to file-based examples
        if not sql_query:
            sql_query, tables = search_examples_for_sql(question, sql_examples)
        
        # Step 4: Generate SQL if no matches
        if not sql_query:
            logger.info("No exact match found; generating SQL as fallback.")
            tables = extract_relevant_tables(question, db)
            table_info = db.get_table_info(tables)
            generation_prompt = PromptTemplate(
                input_variables=["enter", "table_info"],
                template="""
                System: You are an expert SQL data analyst.
                Given an input question, create a syntactically correct {dialect} query.
                Return only the SQL query.
                Question: "{enter}"
                Available tables: {table_info}
                """
            ).format(dialect=db.dialect, enter=question, table_info=table_info)
            sql_generation_chain = generation_prompt | llm | StrOutputParser()
            sql_query = sql_generation_chain.invoke(question)
            tables = extract_tables_from_sql(sql_query)

        # Step 5: Validate SQL using LLM
        if not validate_sql_with_llm(question, sql_query, db, llm):
            return {"response": "The generated SQL query failed validation. Please refine your question or try again later."}

        # Step 6: Execute and format response
        result = execute_sql_query(sql_query, db)
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
    else:
        logger.info(f"Non-database question detected: {question}")
        response = generate_polite_response(question)

    return {"response": response}

def run_interactive():
    config = {"configurable": {"thread_id": "sql_agent_thread"}}
    print("Hi I am SQL Agent. How can I help you today?")
    while True:
        try:
            user_query = input("User: ").strip()
            if not user_query:
                continue
            print(f"User: {user_query}")

            result = sql_agent_workflow.invoke({"question": user_query}, config=config)
            print(f"Assistant: {result['response']}")

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"Assistant: Error: {str(e)}")

if __name__ == "__main__":
    run_interactive()