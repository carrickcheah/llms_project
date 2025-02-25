import os
import re
from typing import Dict, TypedDict, Optional, Union, List
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Define the state type
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    error: Optional[str]

# Load environment variables from .env file
load_dotenv()

def initialize_database() -> SQLDatabase:
    """Initialize and validate the database connection."""
    database_uri = os.getenv("DATABASE_URI")
    if not database_uri:
        raise ValueError("DATABASE_URI is not set in the .env file.")
    
    try:
        db = SQLDatabase.from_uri(database_uri)
        # Test the connection by getting table info
        db.get_usable_table_names()
        return db
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {str(e)}")

def initialize_llm() -> ChatOpenAI:
    """Initialize and validate the language model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the .env file.")
    
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    
    try:
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature)
    except Exception as e:
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")

# Initialize resources
try:
    db = initialize_database()
    llm = initialize_llm()
except Exception as e:
    print(f"Initialization error: {str(e)}")
    exit(1)

# Define the prompt templates
query_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given an input question, create a syntactically correct {dialect} query to run to help find the answer. 
            Unless the user specifies in their question a specific number of examples they wish to obtain, always limit 
            your query to at most {top_k} results. You can order the results by a relevant column to return the most 
            interesting examples in the database.
            
            Never query for all the columns from a specific table, only ask for a few relevant columns given the 
            question.
            
            Pay attention to use only the column names that you can see in the schema description. Be careful to not 
            query for columns that do not exist. Also, pay attention to which column is in which table.
            
            Only use the following tables:
            {table_info}
            
            Question: {input}
            """,
        ),
        ("human", "{input}"),
    ]
)

answer_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given the question, the SQL query used, and the SQL query result, provide a natural language answer.
            
            Format the answer in a clear, concise manner. If the result contains numerical data, provide context and interpretation.
            If the result is empty, explain what that likely means in the context of the question.
            
            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            """,
        ),
    ]
)

def extract_sql_from_response(response_text: str) -> str:
    """Extract SQL query from a markdown-formatted response."""
    # Look for SQL code blocks with or without language specification
    sql_pattern = r"```(?:sql)?\s*(.*?)```"
    matches = re.findall(sql_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return response_text.strip()

def write_query(state: State) -> State:
    """Generate SQL query to fetch information."""
    try:
        # Generate the prompt
        prompt = query_prompt_template.format_messages(
            dialect=db.dialect,
            top_k=10,
            table_info=db.get_table_info(),
            input=state["question"],
        )
        
        # Use the language model to generate the SQL query
        result = llm.invoke(prompt)
        
        # Extract SQL from the response
        raw_query = extract_sql_from_response(result.content)
        
        # Update state
        state["query"] = raw_query
        state["error"] = None
        return state
    except Exception as e:
        state["error"] = f"Failed to generate query: {str(e)}"
        return state

def execute_query(state: State) -> State:
    """Execute the generated SQL query and return the results."""
    if state.get("error"):
        return state
    
    try:
        result = db.run(state["query"])
        state["result"] = result
        state["error"] = None
        return state
    except Exception as e:
        state["error"] = f"Failed to execute query: {str(e)}"
        return state

def generate_answer(state: State) -> State:
    """Generate a natural language answer based on SQL results."""
    if state.get("error"):
        return state
    
    try:
        prompt = answer_prompt_template.format_messages(
            question=state["question"],
            query=state["query"],
            result=state["result"]
        )
        
        result = llm.invoke(prompt)
        state["answer"] = result.content
        return state
    except Exception as e:
        state["error"] = f"Failed to generate answer: {str(e)}"
        return state

def process_question(question: str) -> State:
    """Process a natural language question through the entire pipeline."""
    # Initialize state
    state: State = {
        "question": question,
        "query": "",
        "result": "",
        "answer": "",
        "error": None
    }
    
    # Generate SQL query
    state = write_query(state)
    if state.get("error"):
        return state
    
    # Execute SQL query
    state = execute_query(state)
    if state.get("error"):
        return state
    
    # Generate natural language answer
    state = generate_answer(state)
    return state

def display_results(state: State) -> None:
    """Display the results of the query processing pipeline."""
    print("\n" + "="*80)
    print(f"Question: {state['question']}")
    print("="*80)
    
    if state.get("error"):
        print(f"\nError: {state['error']}")
        return
    
    print("\nGenerated SQL Query:")
    print("-"*40)
    print(state["query"])
    
    print("\nQuery Result:")
    print("-"*40)
    print(state["result"])
    
    print("\nNatural Language Answer:")
    print("-"*40)
    print(state["answer"])
    print("="*80)

def main() -> None:
    """Main function to run the SQL agent."""
    try:
        print(f"Database dialect: {db.dialect}")
        print(f"Available tables: {', '.join(db.get_usable_table_names())}")
        
        while True:
            # Get user input
            print("\nEnter your question (or 'exit' to quit):")
            question = input("> ")
            
            if question.lower() in ["exit", "quit"]:
                break
            
            if not question.strip():
                continue
            
            # Process the question
            state = process_question(question)
            
            # Display results
            display_results(state)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()