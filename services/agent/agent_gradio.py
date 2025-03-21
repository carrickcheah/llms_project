# agent_gradio.py (renamed from main.py for Gradio focus)
import os
import json
import gradio as gr
from loguru import logger
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

# Configure loguru
logger.remove()
logger.add("sql_agent.log", rotation="500 MB", retention="10 days", level="DEBUG")
logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO")

# Custom utility imports
from utils.llm import initialize_openai, initialize_embeddings, initialize_nvidia_deep
from utils.maria import initialize_database
from utils.vector_database import qdrant_on_prem
from utils.tools import load_sql_examples, is_database_question
from utils.task import (
    execute_sql_with_no_data_handling, find_similar_examples, extract_relevant_tables,
    execute_sql_query, validate_sql_with_llm, search_examples_for_sql,
    extract_tables_from_sql, generate_dynamic_sql, format_response
)

# Global initialization (matches your log output)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm = initialize_nvidia_deep(os.getenv("NVIDIA_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

# Global feedback store
feedback_store = {}

def save_feedback_to_file():
    """Save feedback_store to a JSON file locally."""
    try:
        with open("feedback_log.json", "w") as f:
            json.dump(feedback_store, f, indent=2)
        logger.info("Feedback saved to feedback_log.json")
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")

@task
def generate_polite_response(question: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
    You are a friendly SQL Agent. Your role is to assist users with database-related questions.
    If a user asks something unrelated to the database, respond politely and guide them to ask a database-related question.
    User's input: "{question}"
    Your response:
    """)
    response_chain = prompt | llm | StrOutputParser()
    return response_chain.invoke({"question": question})

@task
def collect_user_feedback(question: str, response: str, feedback_store: dict, is_helpful: bool = None, comment: str = None) -> dict:
    try:
        logger.info(f"Collecting feedback for question: '{question}' | Response: '{response[:50]}...'")
        feedback = {"is_helpful": is_helpful, "comment": comment}
        feedback_store.setdefault("responses", []).append({
            "question": question, "response": response,
            "is_helpful": feedback["is_helpful"], "comment": feedback["comment"]
        })
        logger.info(f"Feedback recorded: Helpful={feedback['is_helpful']}, Comment={feedback['comment']}")
        return feedback
    except Exception as e:
        logger.error(f"Error collecting feedback: {str(e)}")
        return {"is_helpful": None, "comment": f"Error: {str(e)}"}

@entrypoint(checkpointer=MemorySaver())
def sql_agent_workflow(inputs: dict, config: dict = None) -> dict:
    question = inputs.get("question", "").strip()
    if not question:
        return {"response": "Please provide a question.", "feedback": None}

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    feedback_store.setdefault(thread_id, {})

    def check_tables_exist(tables, database):
        existing_tables = set(database.get_usable_table_names())
        missing_tables = [t for t in tables if t not in existing_tables]
        return missing_tables

    if is_database_question(question):
        logger.info(f"Database question detected: {question}")
        similar_examples_future = find_similar_examples(question, qdrant_store, embeddings)
        similar_examples = similar_examples_future.result()

        sql_query = None
        tables = []
        if similar_examples:
            for example in similar_examples:
                if example["question"].strip().lower() == question.strip().lower():
                    sql_query = example["sql"]
                    tables = extract_tables_from_sql(sql_query)
                    logger.info(f"Exact match found in Qdrant: '{sql_query}'")
                    break

        if not sql_query:
            sql_query, tables = search_examples_for_sql(question, sql_examples)

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

    feedback_future = collect_user_feedback(question, response, feedback_store[thread_id])
    feedback = feedback_future.result()

    return {"response": response, "feedback": feedback}

def sql_agent_chat(message, history):
    """Wrapper function for Gradio ChatInterface."""
    config = {"configurable": {"thread_id": "sql_agent_thread"}}
    user_input = message["content"] if isinstance(message, dict) else message
    result = sql_agent_workflow.invoke({"question": user_input}, config=config)
    return result["response"]

def main():
    demo = gr.ChatInterface(
        fn=sql_agent_chat,
        type="messages",  # Explicitly set to fix deprecation warning
        title="Nex SQL Agent",
        description="Your friendly SQL data analyst. Ask me anything about the database!",
        examples=[
            "Show me the top 5 customers by sales",
            "Who is our top customer for Cycling Gloves in 2023?",
            "How many orders were placed this year?"
        ],
        theme="soft",
        chatbot=gr.Chatbot(height=750, placeholder="Ask me a database question..."),
        textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7),
        flagging_mode="manual",  # Supported in older versions
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
        save_history=True,
        css="""
            .chatbot {border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
            .message {font-family: 'Arial', sans-serif;}
            #component-0 {max-width: 2500px; margin: 0 auto;}
            .textbox {max-width: 60px !important;}
        """
    )
    
    demo.launch()

if __name__ == "__main__":
    main()