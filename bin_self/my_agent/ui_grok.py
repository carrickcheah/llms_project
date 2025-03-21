import os
import re
import streamlit as st
from typing import Dict, TypedDict, Optional, Union, List
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SQL Agent Chat",
    page_icon="🤖",
    layout="centered",
)

# Define the state type
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    error: Optional[str]

# Initialize session state for database and LLM
if "db" not in st.session_state:
    st.session_state.db = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you query your database using natural language. What would you like to know?"}]
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Prompt templates
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
    sql_pattern = r"```(?:sql)?\s*(.*?)```"
    matches = re.findall(sql_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return response_text.strip()

def initialize_database(database_uri: str) -> Optional[SQLDatabase]:
    """Initialize and validate the database connection."""
    try:
        db = SQLDatabase.from_uri(database_uri)
        # Test the connection by getting table info
        db.get_usable_table_names()
        return db
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None

def initialize_llm(api_key: str, model_name: str, temperature: float) -> Optional[ChatOpenAI]:
    """Initialize and validate the language model."""
    try:
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"Failed to initialize language model: {str(e)}")
        return None

def write_query(db, llm, question: str) -> Dict:
    """Generate SQL query to fetch information."""
    try:
        # Generate the prompt
        prompt = query_prompt_template.format_messages(
            dialect=db.dialect,
            top_k=10,
            table_info=db.get_table_info(),
            input=question,
        )
        
        # Use the language model to generate the SQL query
        result = llm.invoke(prompt)
        
        # Extract SQL from the response
        raw_query = extract_sql_from_response(result.content)
        
        return {"query": raw_query, "error": None}
    except Exception as e:
        return {"query": "", "error": f"Failed to generate query: {str(e)}"}

def execute_query(db, query: str) -> Dict:
    """Execute the generated SQL query and return the results."""
    try:
        result = db.run(query)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": "", "error": f"Failed to execute query: {str(e)}"}

def generate_answer(llm, question: str, query: str, result: str) -> Dict:
    """Generate a natural language answer based on SQL results."""
    try:
        prompt = answer_prompt_template.format_messages(
            question=question,
            query=query,
            result=result
        )
        
        result = llm.invoke(prompt)
        return {"answer": result.content, "error": None}
    except Exception as e:
        return {"answer": "", "error": f"Failed to generate answer: {str(e)}"}

def process_question(db, llm, question: str) -> State:
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
    query_result = write_query(db, llm, question)
    if query_result.get("error"):
        state["error"] = query_result["error"]
        return state
    
    state["query"] = query_result["query"]
    
    # Execute SQL query
    execution_result = execute_query(db, state["query"])
    if execution_result.get("error"):
        state["error"] = execution_result["error"]
        return state
    
    state["result"] = execution_result["result"]
    
    # Generate natural language answer
    answer_result = generate_answer(llm, question, state["query"], state["result"])
    if answer_result.get("error"):
        state["error"] = answer_result["error"]
        return state
    
    state["answer"] = answer_result["answer"]
    return state

# Sidebar for configuration
with st.sidebar:
    st.title("SQL Agent Configuration")
    
    # Database configuration
    st.subheader("Database Settings")
    database_uri = st.text_input("Database URI", os.getenv("DATABASE_URI", ""), type="password")
    
    # OpenAI configuration
    st.subheader("OpenAI Settings")
    api_key = st.text_input("OpenAI API Key", os.getenv("OPENAI_API_KEY", ""), type="password")
    model_name = st.text_input("Model Name", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    
    # Initialize button
    if st.button("Initialize Agent"):
        if not database_uri:
            st.error("Please provide a Database URI")
        elif not api_key:
            st.error("Please provide an OpenAI API Key")
        else:
            with st.spinner("Initializing..."):
                db = initialize_database(database_uri)
                if db:
                    llm = initialize_llm(api_key, model_name, temperature)
                    if llm:
                        st.session_state.db = db
                        st.session_state.llm = llm
                        st.session_state.initialized = True
                        st.success("Agent initialized successfully!")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Database connected successfully. Available tables: {', '.join(db.get_usable_table_names())}"
                        })

# Main chat interface
st.title("🤖 SQL Agent Chat")
st.caption("Query your database using natural language")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "query" in message:
            with st.expander("View SQL Query"):
                st.code(message["query"], language="sql")
        if "result" in message:
            with st.expander("View Raw SQL Results"):
                st.text(message["result"])

# Input for new questions
if prompt := st.chat_input("Ask a question about your database..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check if initialized
    if not st.session_state.initialized:
        with st.chat_message("assistant"):
            st.write("Please initialize the agent first using the sidebar.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Please initialize the agent first using the sidebar."
            })
    else:
        # Process the question
        with st.spinner("Thinking..."):
            state = process_question(st.session_state.db, st.session_state.llm, prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            if state.get("error"):
                st.error(state["error"])
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Error: {state['error']}"
                })
            else:
                st.write(state["answer"])
                with st.expander("View SQL Query"):
                    st.code(state["query"], language="sql")
                with st.expander("View Raw SQL Results"):
                    st.text(state["result"])
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": state["answer"],
                    "query": state["query"],
                    "result": state["result"]
                })