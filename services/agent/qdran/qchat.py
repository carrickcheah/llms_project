import streamlit as st
import time
import json
import ast
import logging
from langchain_community.utilities import SQLDatabase
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database configuration
DATABASE_URI = "mysql+pymysql://toolbox_user:my-password@127.0.0.1:3306/toolbox_db"
db = SQLDatabase.from_uri(DATABASE_URI)

# Qdrant configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hotel"

# Embedding model
embeddings = OpenAIEmbeddings()  # Ensure OPENAI_API_KEY is set

# LLM for chaining (using OpenAI's gpt-4o)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def setup_qdrant():
    """Set up Qdrant vector store from an existing collection."""
    try:
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            url=QDRANT_URL,
            prefer_grpc=True
        )
        return vector_store
    except Exception as e:
        raise Exception(f"Failed to connect to Qdrant collection '{COLLECTION_NAME}': {str(e)}. Ensure the collection exists and is populated.")

def execute_sql_query(sql, params=None):
    """Execute an SQL query on the database with optional parameters."""
    try:
        if params:
            return db.run(sql, parameters=params)
        return db.run(sql)
    except Exception as e:
        raise Exception(f"SQL Execution Error: {str(e)}")

def get_chat_response(messages, prompt, vector_store):
    """Fetch response with RAG and database chaining."""
    logging.info("Fetching chat response...")
    
    # Step 1: Search Qdrant for relevant tools
    results = vector_store.similarity_search(prompt, k=1)
    
    if results:
        logging.info("Found relevant document in Qdrant.")
        doc = results[0]
        tool_name = doc.metadata.get('tool_name', 'Unknown')
        description = doc.page_content.split('Description: ')[1].split('\nSQL: ')[0] if 'Description: ' in doc.page_content else doc.page_content
        sql_template = doc.page_content.split('SQL: ')[1] if 'SQL: ' in doc.page_content else 'No SQL provided'
        
        # Extract parameters from metadata
        params_info = doc.metadata.get('parameters', '[]')
        try:
            params_list = json.loads(params_info) if isinstance(params_info, str) else params_info
        except json.JSONDecodeError:
            params_list = []
        
        # Ensure params_list is a list of dictionaries
        if isinstance(params_list, list) and all(isinstance(p, str) for p in params_list):
            params_list = [{"name": param} for param in params_list]
        
        # Prompt template to extract parameters from user input
        param_prompt = PromptTemplate(
            input_variables=["prompt", "description", "params"],
            template="Given the user prompt '{prompt}' and the tool description '{description}', extract the required parameters {params} from the prompt. Return them as a list in order, or 'None' if not found."
        )
        
        # Use RunnableSequence instead of LLMChain
        param_chain = param_prompt | llm
        
        # Use .invoke() instead of .run()
        param_response = param_chain.invoke({
            "prompt": prompt,
            "description": description,
            "params": [p["name"] for p in params_list]
        })
        
        # Parse parameters from LLM response
        try:
            params = ast.literal_eval(param_response.content) if param_response.content != 'None' else None
        except (ValueError, SyntaxError):
            params = None
        
        # Execute SQL if parameters are provided or not required
        if 'SELECT' in sql_template.upper():
            try:
                if params and isinstance(params, list):
                    sql_result = execute_sql_query(sql_template, tuple(params))
                elif not params_list:  # No parameters needed
                    sql_result = execute_sql_query(sql_template)
                else:
                    sql_result = "Please provide the required parameters."
            except Exception as e:
                sql_result = f"Error executing SQL: {str(e)}"
            
            context = f"Relevant tool: {tool_name}\nDescription: {description}\nSQL: {sql_template}\nDatabase Result: {sql_result}"
        elif 'UPDATE' in sql_template.upper() and params and isinstance(params, list):
            try:
                sql_result = execute_sql_query(sql_template, tuple(params))
            except Exception as e:
                sql_result = f"Error executing SQL: {str(e)}"
            context = f"Relevant tool: {tool_name}\nDescription: {description}\nSQL: {sql_template}\nDatabase Update: {sql_result}"
        else:
            context = f"Relevant tool: {tool_name}\nDescription: {description}\nSQL: {sql_template}\nProvide parameters if needed."
        
        augmented_messages = messages + [{"role": "system", "content": f"Use this context: {context}"}]
    else:
        logging.warning("No relevant document found in Qdrant.")
        augmented_messages = messages
    
    # Step 2: Check if the user wants to list hotels or perform a database query
    if "list hotels" in prompt.lower() or "search hotel" in prompt.lower():
        try:
            # Generate SQL query dynamically using LLM
            sql_prompt = PromptTemplate(
                input_variables=["prompt"],
                template="Generate an SQL query to retrieve hotel data based on the user prompt: '{prompt}'."
            )
            sql_chain = sql_prompt | llm
            sql_query = sql_chain.invoke({"prompt": prompt}).content.strip()
            
            # Execute the generated SQL query
            sql_result = execute_sql_query(sql_query)
            
            # Format the result
            if sql_result:
                hotel_list = "\n".join([f"- {row[0]} ({row[1]})" for row in sql_result])
                response = f"Here are some hotels:\n{hotel_list}"
            else:
                response = "No hotels found."
        except Exception as e:
            response = f"Error querying the database: {str(e)}"
        
        return response
    
    # Step 3: Final response generation
    final_prompt = PromptTemplate(
        input_variables=["messages"],
        template="Based on the conversation history and context, generate a concise and accurate response.\n\nMessages:\n{messages}"
    )
    
    # Chain the final prompt with the LLM
    final_chain = final_prompt | llm
    final_response = final_chain.invoke({"messages": augmented_messages})
    
    return final_response.content

def generate_thought_process(user_input):
    """Generate a simulated thought process for the user's query."""
    base_steps = [
        f"Processing query: '{user_input}'",
        "Breaking down the question to identify key components...",
        "Searching vector store for relevant tools..."
    ]
    
    if "how many" in user_input.lower() and "table" in user_input.lower():
        base_steps.append("Counting tables in the database...")
    elif "sql" in user_input.lower():
        base_steps.append("Executing database query...")
    else:
        base_steps.append("Structuring a comprehensive yet concise response...")
    
    return base_steps

def display_message(role, content):
    """Helper function to display chat messages with appropriate styling."""
    message_class = "user-message" if role == "user" else "assistant-message"
    with st.chat_message(role):
        st.markdown(f'<div class="{message_class}">{content}</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Nex Ai Agent Chat")
    
    try:
        db = SQLDatabase.from_uri(DATABASE_URI)
        st.write("Connected to the database! Yay!")
    except Exception as e:
        st.write(f"Oh no! Couldn’t connect: {str(e)}")
        return
    
    try:
        vector_store = setup_qdrant()
        st.write("Qdrant vector store initialized!")
    except Exception as e:
        st.write(f"{str(e)}")
        return
    
    st.markdown("""
        <style>
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
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Nex AI Agent")
    st.caption(f"Powered by GPT-4o with Qdrant RAG and DB | Date: {datetime.now().strftime('%B %d, %Y')}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI assistant with hotel management tools and database access. How can I assist you today?"}
        ]
    
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])
    
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)
        
        thought_placeholder = st.empty()
        thought_steps = generate_thought_process(prompt)
        
        for step in thought_steps:
            with thought_placeholder.chat_message("assistant"):
                st.markdown(f'<div class="thought-message">🤔 {step}</div>', unsafe_allow_html=True)
            time.sleep(0.8)
        
        with st.spinner("Finalizing response..."):
            if "how many" in prompt.lower() and "table" in prompt.lower():
                tables = db.get_usable_table_names()
                response = f"There are {len(tables)} tables in the database. Here they are:\n\n{', '.join(tables)}"
            else:
                response = get_chat_response(st.session_state.messages, prompt, vector_store)
            
            full_response = (
                f"**Thought Process:**\n\n" + "\n".join(thought_steps) +
                f"\n\n**Answer:**\n\n{response}"
            )
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            display_message("assistant", full_response)

if __name__ == "__main__":
    main()