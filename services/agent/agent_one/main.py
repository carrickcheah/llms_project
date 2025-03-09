# main.py ( always keep this # comment)

import os
from loguru import logger
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

# Custom utility imports
from model import initialize_openai, initialize_embeddings
from maria import initialize_database
from vectordb import qdrant_on_prem
from task_03 import load_sql_examples, is_database_question
from task_01 import (
    find_sql_examples,
    extract_relevant_tables,
    validate_sql_with_llm,
    extract_tables_from_sql,
    generate_dynamic_sql,
    check_tables_exist,
)
from task_02 import (
    execute_sql_with_no_data_handling,
    collect_user_feedback,
    generate_response,
)

# Global initialization (Never edit this part)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm = initialize_openai(os.getenv("OPENAI_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

@entrypoint(checkpointer=MemorySaver())
def sql_agent_workflow(inputs: dict, config: dict = None) -> dict:
    """
    Workflow to process a user's question, generate and execute an SQL query if applicable,
    and return a formatted response with feedback collection for database questions only.
    
    Args:
        inputs (dict): Dictionary containing the user's question under the key "question".
        config (dict): Runtime configuration (e.g., thread_id for feedback store)
    
    Returns:
        dict: Dictionary with "response" and "feedback" keys (feedback is None for non-database questions)
    """
    question = inputs.get("question", "").strip()
    if not question:
        return {"response": "Please provide a question.", "feedback": None}

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    feedback_store = globals().setdefault("feedback_store", {}).setdefault(thread_id, {})

    if is_database_question(question):
        logger.info(f"Database question detected: {question}")

        # Step 1: Search for examples (vector + exact match)
        examples_future = find_sql_examples(question, qdrant_store, embeddings, sql_examples, method="both")
        examples = examples_future.result()
        sql_query = None
        tables = []
        exact_match_found = False
        if examples:
            for example in examples:
                if example["question"].strip().lower() == question.strip().lower():
                    sql_query = example["sql"]
                    tables = example["tables"]
                    logger.info(f"Exact match found: '{sql_query}'")
                    exact_match_found = True
                    break

        # Step 2: Validate and execute if exact match found, otherwise generate new SQL
        if sql_query and exact_match_found:
            missing_tables = check_tables_exist(tables, db)
            if missing_tables:
                logger.error(f"Table(s) {missing_tables} not found in database.")
                response = f"Sorry, I can't execute the query because the following table(s) were not found: {', '.join(missing_tables)}."
            else:
                is_valid_future = validate_sql_with_llm(question, sql_query, db, llm)
                is_valid = is_valid_future.result()
                if not is_valid:
                    logger.info("Exact match SQL failed validation; falling back to generation.")
                    sql_query = None  # Reset to trigger generation
                else:
                    response = execute_sql_with_no_data_handling(question, sql_query, db, llm)

        # Step 3: Generate SQL if no valid exact match
        if not sql_query:
            logger.info("No valid exact match found; generating SQL.")
            tables = extract_relevant_tables(question, db)
            missing_tables = check_tables_exist(tables, db)
            if missing_tables:
                logger.error(f"Table(s) {missing_tables} not found in database.")
                response = f"Sorry, I can't process your request because the following table(s) were not found: {', '.join(missing_tables)}."
            else:
                sql_query_future = generate_dynamic_sql(question, tables, examples or [], db, llm)
                sql_query = sql_query_future.result()
                if sql_query.startswith("Error:"):
                    logger.error(f"SQL generation failed: {sql_query}")
                    response = f"Sorry, I couldn’t generate an SQL query due to an error: {sql_query}"
                else:
                    tables = extract_tables_from_sql(sql_query)
                    missing_tables = check_tables_exist(tables, db)
                    if missing_tables:
                        logger.error(f"Table(s) {missing_tables} not found in database.")
                        response = f"Sorry, I can't execute the query because the following table(s) were not found: {', '.join(missing_tables)}."
                    else:
                        is_valid_future = validate_sql_with_llm(question, sql_query, db, llm)
                        is_valid = is_valid_future.result()
                        if not is_valid:
                            response = "The generated SQL query failed validation. Please try rephrasing your question (e.g., check product name or year)."
                        else:
                            response = execute_sql_with_no_data_handling(question, sql_query, db, llm)

        if not sql_query:
            response = "Sorry, I could not generate a suitable SQL query for your question. Please try rephrasing it."

        # Step 4: Collect feedback only for database questions
        feedback_future = collect_user_feedback(question, response, feedback_store)
        feedback = feedback_future.result()

    else:
        logger.info(f"Non-database question detected: {question}")
        response_future = generate_response(question, llm, response_type="polite")
        response = response_future.result()
        feedback = None  # No feedback for non-database questions

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

            # Collect feedback interactively only if it's a database question
            if is_database_question(user_query):
                feedback_response = input("Was this answer helpful? (Yes/No): ").strip().lower()
                feedback_comment = input("Any comments? (Press Enter to skip): ").strip() or None
                is_helpful = True if feedback_response == "yes" else False if feedback_response == "no" else None
                
                thread_id = config["configurable"]["thread_id"]
                feedback_store = globals()["feedback_store"][thread_id]
                if feedback_store.get("responses"):
                    last_entry = feedback_store["responses"][-1]
                    last_entry["is_helpful"] = is_helpful
                    last_entry["comment"] = feedback_comment
                    logger.info(f"Feedback recorded: Helpful={is_helpful}, Comment={feedback_comment}")
                    print(f"Feedback: Helpful={'Yes' if is_helpful else 'No' if is_helpful is False else 'N/A'}, Comment={feedback_comment or 'None'}")
                else:
                    logger.warning("No responses in feedback store to update.")

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"Assistant: Error: {str(e)}")

if __name__ == "__main__":
    run_interactive()