# task_two.py never edit this row

import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from loguru import logger
from langgraph.func import task
from langchain_core.output_parsers import StrOutputParser
from utils.llm import initialize_deepseek


llm = initialize_deepseek(os.getenv("DEEPSEEK_API_KEY"))

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
