# main.py ( always keep this # comment )
import os
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
from utils.task import execute_sql_with_no_data_handling
from utils.task import (
    find_similar_examples,
    extract_relevant_tables,
    execute_sql_query,
    validate_sql_with_llm,
    search_examples_for_sql,
    extract_tables_from_sql,
)

# Global initialization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm = initialize_openai(os.getenv("OPENAI_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

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
    """
    Workflow to process a user's question, generate and execute an SQL query if applicable,
    and return a formatted response. Handles all no-data cases gracefully.
    
    Args:
        inputs (dict): Dictionary containing the user's question under the key "question".
    
    Returns:
        dict: Dictionary with a "response" key containing the answer or a no-data message.
    """
    question = inputs.get("question", "").strip()
    if not question:
        return {"response": "Please provide a question."}

    # Helper function to check if all required tables exist in the database
    def check_tables_exist(tables, database):
        existing_tables = set(database.get_usable_table_names())
        missing_tables = [t for t in tables if t not in existing_tables]
        return missing_tables

    if is_database_question(question):
        logger.info(f"Database question detected: {question}")

        # Step 1: Search for similar examples in vector store
        similar_examples = find_similar_examples(question, qdrant_store, embeddings)

        # Step 2: Check for an exact match in vector search results
        sql_query = None
        tables = []
        if similar_examples:
            for example in similar_examples:
                if example["question"].strip().lower() == question.strip().lower():
                    sql_query = example["sql"]
                    tables = extract_tables_from_sql(sql_query)
                    logger.info(f"Exact match found in Qdrant: '{sql_query}'")
                    break

        # Step 3: Fallback to file-based examples if no exact match
        if not sql_query:
            sql_query, tables = search_examples_for_sql(question, sql_examples)

        # Step 4: Generate SQL if no matches are found
        if not sql_query:
            logger.info("No exact match found; generating SQL as fallback.")
            tables = extract_relevant_tables(question, db)
            missing_tables = check_tables_exist(tables, db)
            if missing_tables:
                logger.error(f"Table(s) {missing_tables} not found in database.")
                return {
                    "response": f"Sorry, I can't process your request because the following table(s) were not found: {', '.join(missing_tables)}."
                }
            table_info = db.get_table_info(tables)
            generation_prompt = PromptTemplate(
                input_variables=["dialect", "question", "table_info"],
                template="""
                System: You are an expert SQL data analyst.
                Given an input question, create a syntactically correct {dialect} query.
                Return only the SQL query.
                Question: "{question}"
                Available tables: {table_info}
                """
            )
            sql_generation_chain = generation_prompt | llm | StrOutputParser()
            sql_query = sql_generation_chain.invoke({
                "dialect": db.dialect,
                "question": question,
                "table_info": table_info
            })
            tables = extract_tables_from_sql(sql_query)

        # Step 5: Validate the generated SQL query
        if sql_query is None or sql_query == "":
            logger.warning("Failed to generate SQL query.")
            return {"response": "Sorry, I could not generate a suitable SQL query for your question. Please try rephrasing it."}
        
        missing_tables = check_tables_exist(tables, db)
        if missing_tables:
            logger.error(f"Table(s) {missing_tables} not found in database.")
            return {
                "response": f"Sorry, I can't execute the query because the following table(s) were not found: {', '.join(missing_tables)}."
            }

        if not validate_sql_with_llm(question, sql_query, db, llm):
            return {"response": "The generated SQL query failed validation. Please refine your question or try again later."}

        # Step 6: Execute the query with no-data handling
        response = execute_sql_with_no_data_handling(question, sql_query, db, llm)
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