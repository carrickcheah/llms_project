# sql_agent_graph.py

import os
from loguru import logger
from typing import TypedDict, Literal, Optional, Dict, Union, List, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.func import EntryPoint, task
from langgraph.checkpoint.memory import MemorySaver

# Import from existing code
from model import initialize_openai, initialize_embeddings, initialize_open_deep, initialize_nvidia_deep
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

# Setup environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize global components
llm = initialize_nvidia_deep(os.getenv("NVIDIA_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

# Define a state class for the graph
class State(TypedDict):
    question: str
    is_db_question: bool
    examples: Optional[List[Dict]]
    tables: Optional[List[str]]
    sql_query: Optional[str]
    missing_tables: Optional[List[str]]
    is_valid_sql: Optional[bool]
    response: Optional[str]
    feedback: Optional[Dict]
    error: Optional[str]

# Node functions for the graph
@task
def determine_question_type(state: State) -> State:
    """Determine if the question is a database-related question."""
    question = state["question"]
    is_db_question = is_database_question(question)
    logger.info(f"Question type: {'Database' if is_db_question else 'Non-database'}")
    return {**state, "is_db_question": is_db_question}

@task
def handle_non_db_question(state: State) -> State:
    """Generate a polite response for non-database questions."""
    response = generate_response(
        question=state["question"],
        llm=llm,
        response_type="polite"
    )
    return {**state, "response": response, "feedback": None}

@task
def search_examples(state: State) -> State:
    """Search for similar SQL examples."""
    examples = find_sql_examples(
        question=state["question"],
        vector_store=qdrant_store,
        embeddings=embeddings,
        sql_examples=sql_examples,
        method="both"
    )
    return {**state, "examples": examples or []}

@task
def check_exact_match(state: State) -> State:
    """Check for an exact match in examples."""
    question = state["question"]
    examples = state["examples"]
    
    for example in examples:
        if example["question"].strip().lower() == question.strip().lower():
            logger.info(f"Exact match found: '{example['sql']}'")
            sql_query = example["sql"]
            tables = example["tables"]
            
            # Check if all tables exist
            missing_tables = check_tables_exist(tables, db)
            
            return {
                **state, 
                "sql_query": sql_query, 
                "tables": tables,
                "missing_tables": missing_tables,
                "is_exact_match": True
            }
    
    return {**state, "is_exact_match": False}

@task
def identify_tables(state: State) -> State:
    """Identify relevant tables for the question."""
    tables = extract_relevant_tables(state["question"], db)
    missing_tables = check_tables_exist(tables, db)
    return {**state, "tables": tables, "missing_tables": missing_tables}

@task
def generate_sql(state: State) -> State:
    """Generate a dynamic SQL query."""
    question = state["question"]
    tables = state["tables"]
    examples = state["examples"]
    
    sql_query = generate_dynamic_sql(
        question=question,
        relevant_tables=tables,
        similar_examples=examples,
        db=db,
        llm=llm
    )
    
    if sql_query.startswith("Error:"):
        return {**state, "error": sql_query}
    
    # Extract tables from the generated SQL
    extracted_tables = extract_tables_from_sql(sql_query)
    missing_tables = check_tables_exist(extracted_tables, db)
    
    return {
        **state, 
        "sql_query": sql_query, 
        "tables": extracted_tables,
        "missing_tables": missing_tables
    }

@task
def validate_sql(state: State) -> State:
    """Validate the SQL query."""
    # Skip validation for exact matches
    if state.get("is_exact_match", False):
        return {**state, "is_valid_sql": True}
    
    is_valid = validate_sql_with_llm(
        question=state["question"],
        sql_query=state["sql_query"],
        db=db,
        llm=llm
    )
    
    return {**state, "is_valid_sql": is_valid}

@task
def execute_sql(state: State) -> State:
    """Execute the SQL query and generate a response."""
    response = execute_sql_with_no_data_handling(
        question=state["question"],
        sql_query=state["sql_query"],
        db=db,
        llm=llm
    )
    
    return {**state, "response": response}

@task
def handle_missing_tables(state: State) -> State:
    """Generate a response when tables are missing."""
    missing_tables = state["missing_tables"]
    response = f"Sorry, I can't execute the query because the following table(s) were not found: {', '.join(missing_tables)}."
    return {**state, "response": response}

@task
def handle_invalid_sql(state: State) -> State:
    """Generate a response when SQL is invalid."""
    examples = state["examples"]
    tables = state["tables"]
    
    products_list = [ex["question"] for ex in examples[:5]] if examples else []
    error_context = {
        "question": state["question"],
        "available_examples": products_list,
        "database_tables": tables
    }
    
    response = generate_response(
        question=state["question"],
        llm=llm,
        response_type="validation_failure",
        context=error_context
    )
    
    return {**state, "response": response}

@task
def handle_error(state: State) -> State:
    """Generate a response when an error occurs."""
    error = state["error"]
    response = f"Sorry, I couldn't generate an SQL query due to an error: {error}"
    return {**state, "response": response}

@task
def collect_feedback(state: State) -> State:
    """Collect user feedback for the response."""
    # In a real application, this would be a placeholder for actual feedback collection
    thread_id = "default_thread"  # In practice, get this from config
    feedback_store = {}
    
    feedback = collect_user_feedback(
        question=state["question"],
        response=state["response"],
        feedback_store=feedback_store
    )
    
    return {**state, "feedback": feedback}

@task
def prepare_output(state: State) -> Dict:
    """Prepare the final output."""
    return {
        "response": state["response"],
        "feedback": state.get("feedback")
    }

# Define conditional routing functions
def route_by_question_type(state: State) -> Literal["handle_db_question", "handle_non_db_question"]:
    """Route based on whether it's a database question."""
    return "handle_db_question" if state["is_db_question"] else "handle_non_db_question"

def route_after_examples(state: State) -> Literal["check_exact_match", "identify_tables"]:
    """Route after finding examples."""
    return "check_exact_match" if state["examples"] else "identify_tables"

def route_after_exact_match(state: State) -> Literal["execute_tables_check", "identify_tables"]:
    """Route after checking for exact matches."""
    return "execute_tables_check" if state.get("is_exact_match", False) else "identify_tables"

def route_tables_check(state: State) -> Literal["handle_missing_tables", "generate_sql", "execute_sql"]:
    """Route after checking tables."""
    if state["missing_tables"]:
        return "handle_missing_tables"
    
    if state.get("is_exact_match", False):
        # For exact matches, skip SQL generation and validation
        return "execute_sql"
    
    return "generate_sql"

def route_after_sql_generation(state: State) -> Literal["handle_error", "handle_missing_tables", "validate_sql"]:
    """Route after SQL generation."""
    if "error" in state and state["error"]:
        return "handle_error"
    
    if state["missing_tables"]:
        return "handle_missing_tables"
    
    return "validate_sql"

def route_after_validation(state: State) -> Literal["handle_invalid_sql", "execute_sql"]:
    """Route after SQL validation."""
    return "execute_sql" if state["is_valid_sql"] else "handle_invalid_sql"

# Create the graph
def build_sql_agent_graph():
    # Initialize state graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("determine_question_type", determine_question_type)
    graph.add_node("handle_non_db_question", handle_non_db_question)
    graph.add_node("search_examples", search_examples)
    graph.add_node("check_exact_match", check_exact_match)
    graph.add_node("identify_tables", identify_tables)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("handle_missing_tables", handle_missing_tables)
    graph.add_node("handle_invalid_sql", handle_invalid_sql)
    graph.add_node("handle_error", handle_error)
    graph.add_node("collect_feedback", collect_feedback)
    graph.add_node("prepare_output", prepare_output)
    
    # Add edges
    graph.set_entry_point("determine_question_type")
    
    # Routing based on question type
    graph.add_conditional_edges(
        "determine_question_type",
        route_by_question_type,
        {
            "handle_db_question": "search_examples",
            "handle_non_db_question": "handle_non_db_question"
        }
    )
    
    graph.add_edge("handle_non_db_question", "prepare_output")
    
    # Database question handling flow
    graph.add_conditional_edges(
        "search_examples",
        route_after_examples,
        {
            "check_exact_match": "check_exact_match",
            "identify_tables": "identify_tables"
        }
    )
    
    graph.add_conditional_edges(
        "check_exact_match",
        route_after_exact_match,
        {
            "execute_tables_check": "execute_tables_check",
            "identify_tables": "identify_tables"
        }
    )
    
    graph.add_conditional_edges(
        "identify_tables",
        route_tables_check,
        {
            "handle_missing_tables": "handle_missing_tables",
            "generate_sql": "generate_sql",
            "execute_sql": "execute_sql"
        }
    )
    
    graph.add_conditional_edges(
        "generate_sql",
        route_after_sql_generation,
        {
            "handle_error": "handle_error",
            "handle_missing_tables": "handle_missing_tables",
            "validate_sql": "validate_sql"
        }
    )
    
    graph.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "handle_invalid_sql": "handle_invalid_sql",
            "execute_sql": "execute_sql"
        }
    )
    
    # Connect all response generators to feedback collection
    graph.add_edge("execute_sql", "collect_feedback")
    graph.add_edge("handle_missing_tables", "collect_feedback")
    graph.add_edge("handle_invalid_sql", "collect_feedback")
    graph.add_edge("handle_error", "collect_feedback")
    
    # Final output preparation
    graph.add_edge("collect_feedback", "prepare_output")
    graph.add_edge("prepare_output", END)
    
    # Compile the graph
    return graph.compile(checkpointer=MemorySaver())

# Create the entrypoint
@EntryPoint(build_sql_agent_graph())
def sql_agent(question: str, config: dict = None) -> dict:
    """
    LangGraph entrypoint for SQL agent.
    
    Args:
        question (str): The user's question
        config (dict): Configuration options
        
    Returns:
        dict: Response and feedback
    """
    if not question.strip():
        return {"response": "Please provide a question.", "feedback": None}
    
    # Initialize the state
    initial_state = {"question": question}
    
    # Return the result
    return initial_state

# Interactive mode
def run_interactive():
    config = {"configurable": {"thread_id": "sql_agent_thread"}}
    print("Hi I am SQL Agent. How can I help you today?")
    while True:
        try:
            user_query = input("User: ").strip()
            if not user_query:
                continue
                
            print(f"User: {user_query}")
            result = sql_agent(user_query, config=config)
            print(f"Assistant: {result['response']}")
            
            # Interactive feedback collection
            if is_database_question(user_query):
                feedback_response = input("Was this answer helpful? (Yes/No): ").strip().lower()
                feedback_comment = input("Any comments? (Press Enter to skip): ").strip() or None
                print(f"Feedback recorded: Helpful={'Yes' if feedback_response == 'yes' else 'No' if feedback_response == 'no' else 'N/A'}, Comment={feedback_comment or 'None'}")
                
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"Assistant: Error: {str(e)}")

if __name__ == "__main__":
    run_interactive()