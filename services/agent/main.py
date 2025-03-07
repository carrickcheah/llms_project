# main.py ( always keep this # comment )
import os
from loguru import logger
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

# Custom utility imports
from utils.llm import initialize_openai, initialize_embeddings
from utils.maria import initialize_database
from utils.vector_database import qdrant_on_prem
from utils.tools import load_sql_examples, is_database_question
from utils.task import (
    execute_sql_with_no_data_handling,
    find_similar_examples,
    extract_relevant_tables,
    execute_sql_query,
    validate_sql_with_llm,
    search_examples_for_sql,
    extract_tables_from_sql,
    generate_dynamic_sql,
    format_response,
)

# Global initialization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm = initialize_openai(os.getenv("OPENAI_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

@task
def generate_polite_response(question: str) -> str:
    """
    Generate a polite response for non-database questions.

    Args:
        question (str): The user's input question
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
        # Feedback will be populated in run_interactive; store it here
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

@entrypoint(checkpointer=MemorySaver())
def sql_agent_workflow(inputs: dict, config: dict = None) -> dict:
    """
    Workflow to process a user's question, generate and execute an SQL query if applicable,
    and return a formatted response with feedback collection.
    
    Args:
        inputs (dict): Dictionary containing the user's question under the key "question".
        config (dict): Runtime configuration (e.g., thread_id for feedback store)
    
    Returns:
        dict: Dictionary with "response" and "feedback" keys
    """
    question = inputs.get("question", "").strip()
    if not question:
        return {"response": "Please provide a question.", "feedback": None}

    # Initialize feedback store (thread-specific via config)
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    feedback_store = globals().setdefault("feedback_store", {}).setdefault(thread_id, {})

    # Helper function to check if all required tables exist in the database
    def check_tables_exist(tables, database):
        existing_tables = set(database.get_usable_table_names())
        missing_tables = [t for t in tables if t not in existing_tables]
        return missing_tables

    if is_database_question(question):
        logger.info(f"Database question detected: {question}")

        # Step 1: Search for similar examples in vector store
        similar_examples_future = find_similar_examples(question, qdrant_store, embeddings)
        similar_examples = similar_examples_future.result()

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
                response = f"Sorry, I can't process your request because the following table(s) were not found: {', '.join(missing_tables)}."
            else:
                sql_query_future = generate_dynamic_sql(question, tables, similar_examples, db, llm)
                sql_query = sql_query_future.result()
                if sql_query.startswith("Error:"):
                    logger.error(f"SQL generation failed: {sql_query}")
                    response = f"Sorry, I couldn’t generate an SQL query due to an error: {sql_query}"
                else:
                    tables = extract_tables_from_sql(sql_query)

        # Step 5: Validate and execute the generated SQL query
        if sql_query and not sql_query.startswith("Error:"):
            missing_tables = check_tables_exist(tables, db)
            if missing_tables:
                logger.error(f"Table(s) {missing_tables} not found in database.")
                response = f"Sorry, I can't execute the query because the following table(s) were not found: {', '.join(missing_tables)}."
            else:
                validation_future = validate_sql_with_llm(question, sql_query, db, llm)
                if not validation_future.result():
                    response = "The generated SQL query failed validation. Please refine your question or try again later."
                else:
                    response = execute_sql_with_no_data_handling(question, sql_query, db, llm)
        else:
            if not sql_query:
                response = "Sorry, I could not generate a suitable SQL query for your question. Please try rephrasing it."

    else:
        logger.info(f"Non-database question detected: {question}")
        response_future = generate_polite_response(question)
        response = response_future.result()

    # Step 6: Collect feedback
    feedback_future = collect_user_feedback(question, response, feedback_store)
    feedback = feedback_future.result()

    return {"response": response, "feedback": feedback}

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

            # Collect feedback interactively
            feedback_response = input("Was this answer helpful? (Yes/No): ").strip().lower()
            feedback_comment = input("Any comments? (Press Enter to skip): ").strip() or None
            is_helpful = True if feedback_response == "yes" else False if feedback_response == "no" else None
            
            # Update feedback store with user input
            thread_id = config["configurable"]["thread_id"]
            feedback_store = globals()["feedback_store"][thread_id]
            last_entry = feedback_store["responses"][-1]
            last_entry["is_helpful"] = is_helpful
            last_entry["comment"] = feedback_comment
            logger.info(f"Feedback recorded: Helpful={is_helpful}, Comment={feedback_comment}")

            # Optional: Print feedback summary (for debugging)
            print(f"Feedback: Helpful={'Yes' if is_helpful else 'No' if is_helpful is False else 'N/A'}, Comment={feedback_comment or 'None'}")

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"Assistant: Error: {str(e)}")

if __name__ == "__main__":
    run_interactive()