import streamlit as st
import logging
import re
from datetime import datetime
import time
from typing import List, Dict

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Constants
DATABASE_URI = "mysql+pymysql://toolbox_user:my-password@127.0.0.1:3306/toolbox_db"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "hotela"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS styling
CHAT_CSS = """
    .stChatMessage {
        max-width: 80%;
        margin: 10px;
    }
    .user-message {
        background-color: #2b313e;
        border-radius: 10px;
        padding: 10px;
        animation: fadeIn 0.5s ease-in-out;
    }
    .assistant-message {
        background-color: #1a1e26;
        border-radius: 10px;
        padding: 10px;
        animation: fadeIn 0.5s ease-in-out;
    }
    .thought-message {
        background-color: #3a3f4a;
        border-radius: 10px;
        padding: 10px;
        color: #ffffff;
        font-style: italic;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
"""

@st.cache_resource
def initialize_dependencies():
    """Initialize database, LLM, and vector store with caching"""
    try:
        # Database setup
        db = SQLDatabase.from_uri(DATABASE_URI)
        
        # LLM setup
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
        # Toolkit and embeddings
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        embeddings = OpenAIEmbeddings()
        
        # Qdrant setup
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION,
            url=QDRANT_URL
        )
        
        return db, llm, toolkit, vector_store
    
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}")
        st.error("Error initializing application. Please check logs.")
        raise

def create_prompt_template():
    """Create a prompt template for generating SQL queries"""
    template = """
    You are an expert SQL assistant. Given a user query about hotels and the following context from a vector store, generate an appropriate SQL query or response.

    User Query: {query}
    Context from Vector Store: {context}

    The context contains a tool definition with a user query and a corresponding generated SQL query. If the user's query matches the context's user query, use the same SQL query. If it's similar, adjust the SQL query accordingly.

    First, check if the user's query matches the context's user query. If it does, set your SQL query to be the same as the context's generated SQL query.

    If not, modify the context's generated SQL query to fit the user's query.

    Provide the response in the following exact format:

    SQL Query: <the SQL query you generate>

    Explanation: <a brief explanation of what the query does>

    Do not deviate from this format. The output must have two lines: one starting with "SQL Query:" followed by the query, and another starting with "Explanation:" followed by the explanation.

    Example:
    Context:
    Operation: search-hotels-by-location
    User Query: Find available hotels in Zurich
    Generated SQL: SELECT * FROM hotels WHERE location = 'Zurich' AND booked = 0;
    Query Type: SELECT
    Explanation: Retrieves all available hotels in Zurich by filtering location and booked status.

    User Query: Find available hotels in Paris

    Your task is to modify the SQL query to replace 'Zurich' with 'Paris'.

    So, SQL Query: SELECT * FROM hotels WHERE location = 'Paris' AND booked = 0;

    Explanation: Retrieves all available hotels in Paris by filtering location and booked status.

    Another example:
    User Query: Find available hotels

    If the context has a tool for "search-available-hotels" with SQL "SELECT * FROM hotels WHERE booked = 0;", use that.

    So, SQL Query: SELECT * FROM hotels WHERE booked = 0;

    Explanation: Retrieves all available hotels.

    If the user query is vague and no direct match is found, use a default query like "SELECT * FROM hotels;" to list all hotels.

    So, based on the user's query and the context, generate the appropriate SQL query and explanation.
    """
    return PromptTemplate.from_template(template)

def extract_location_from_select(sql_query: str) -> str:
    """Extract location from a SELECT query if present"""
    pattern = r"location\s*=\s*['\"]([^\"']+)['\"]"
    match = re.search(pattern, sql_query, re.IGNORECASE)
    return match.group(1) if match else None

def extract_hotel_name_from_update(sql_query: str) -> str:
    """Extract hotel name from an UPDATE query if present"""
    pattern = r"name\s*=\s*['\"]([^\"']+)['\"]"
    match = re.search(pattern, sql_query, re.IGNORECASE)
    return match.group(1) if match else None

def parse_select_result(result: str) -> List[tuple]:
    """Parse the string result from db.run into a list of tuples with name and location"""
    if not result or result == "[]":
        return []
    
    # Pattern to match tuples in the result string
    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, result)
    
    parsed_result = []
    for match in matches:
        # Split the tuple string by commas and strip quotes
        parts = [part.strip().strip("'\"") for part in match.split(",")]
        try:
            # Assuming order: id, name, location, price_tier, checkin_date, checkout_date, booked
            name = parts[1]  # Hotel name
            location = parts[2]  # Location
            parsed_result.append((name, location))
        except IndexError:
            logger.error(f"Failed to parse tuple: {match}")
            continue
    
    return parsed_result

def format_select_result(sql_query: str, result: str) -> str:
    """Format SELECT query results into a user-friendly string"""
    parsed_result = parse_select_result(result)
    
    if not parsed_result:
        return "No hotels found."
    
    location = extract_location_from_select(sql_query)
    hotel_names = [row[0] for row in parsed_result]
    
    if location and all(row[1] == location for row in parsed_result):
        return f"Available hotels in {location}: {', '.join(hotel_names)}"
    else:
        hotel_list = [f"{name} in {row[1]}" for name, row in zip(hotel_names, parsed_result)]
        return f"Available hotels: {', '.join(hotel_list)}"

def format_update_result(sql_query: str, result: str) -> str:
    """Format UPDATE query results into a user-friendly string"""
    hotel_name = extract_hotel_name_from_update(sql_query)
    if not hotel_name:
        return result if result.startswith("Error") else f"Operation completed: {result}"
    
    if "booked = 1" in sql_query.lower():
        action = "booked"
    elif "booked = 0" in sql_query.lower():
        action = "canceled"
    else:
        action = "updated"
    
    if result.startswith("Error"):
        return f"Failed to {action} hotel {hotel_name}: {result}"
    return f"Hotel {hotel_name} has been {action}."

def process_query(query: str, vector_store, llm, db) -> Dict[str, str]:
    """Process the user query and return a response"""
    # Search vector store for relevant tools
    search_results = vector_store.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in search_results])
    
    # Create and run the prompt
    prompt = create_prompt_template()
    formatted_prompt = prompt.format(query=query, context=context)
    logger.info(f"Formatted Prompt: {formatted_prompt}")
    
    chain = RunnableSequence(prompt | llm)
    response = chain.invoke({"query": query, "context": context})
    logger.info(f"LLM Response: {response.content}")
    
    # Parse the response with robust handling
    lines = response.content.split("\n")
    sql_query = None
    explanation = None
    for line in lines:
        line = line.strip()
        if line.startswith("SQL Query:"):
            sql_query = line.split(": ", 1)[1]
        elif line.startswith("Explanation:"):
            explanation = line.split(": ", 1)[1]
    
    if not sql_query or "SQL Query" not in response.content:
        sql_query = "No query generated"
        explanation = "No explanation provided"
    
    # Execute the query if it's a valid SQL statement
    if "SELECT" in sql_query.upper():
        try:
            result = db.run(sql_query)
            return {"sql_query": sql_query, "explanation": explanation, "result": result}
        except Exception as e:
            return {"sql_query": sql_query, "explanation": explanation, "result": f"Error: {e}"}
    elif "UPDATE" in sql_query.upper():
        try:
            db.run(sql_query)
            return {"sql_query": sql_query, "explanation": explanation, "result": "Operation completed"}
        except Exception as e:
            return {"sql_query": sql_query, "explanation": explanation, "result": f"Error: {e}"}
    return {"sql_query": sql_query, "explanation": explanation, "result": "No execution required"}

def format_response(response: Dict[str, str], query: str) -> str:
    """Format the response into a user-friendly string"""
    if response["sql_query"] == "No query generated":
        return "Sorry, I didn't understand your query. Please be more specific."
    elif "SELECT" in response["sql_query"].upper():
        return format_select_result(response["sql_query"], response["result"])
    elif "UPDATE" in response["sql_query"].upper():
        return format_update_result(response["sql_query"], response["result"])
    return response["result"]

def main():
    """Main Streamlit app"""
    st.title("Hotel Booking Assistant")
    st.write("Ask about hotel availability or book a hotel!")

    # Inject CSS
    st.markdown(f"<style>{CHAT_CSS}</style>", unsafe_allow_html=True)

    # Initialize dependencies
    db, llm, toolkit, vector_store = initialize_dependencies()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="stChatMessage user-message">{message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f'<div class="stChatMessage assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    # User input
    if query := st.chat_input("What would you like to know or do?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(f'<div class="stChatMessage user-message">{query}</div>', unsafe_allow_html=True)

        # Process query
        with st.spinner("Thinking..."):
            response = process_query(query, vector_store, llm, db)
        
        # Format and display response
        response_text = format_response(response, query)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.markdown(f'<div class="stChatMessage assistant-message">{response_text}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()