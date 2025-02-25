import streamlit as st
from datetime import datetime
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage  # Import message classes

# Database configuration
DATABASE_URI = "mysql+pymysql://myuser:mypassword@127.0.0.1:3306/nex_valiant"

# Try connecting to the database
try:
    db = SQLDatabase.from_uri(DATABASE_URI)
    st.write("Connected to the database! Yay!")
except Exception as e:
    st.write(f"Oh no! Couldn’t connect: {str(e)}")
    st.stop()  # Stop if we can’t connect

# Initialize the local ChatOllama model
model = ChatOllama(model="llama3.1", temperature=0)

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by a local model")

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display existing conversation messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
if prompt := st.chat_input():
    # Append user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Check if the prompt asks about the number of tables
    if "how many tables" in prompt.lower():
        try:
            # Query the information_schema for the table count in the 'nex_valiant' schema.
            query = (
                "SELECT COUNT(*) AS table_count FROM information_schema.tables "
                "WHERE table_schema = 'nex_valiant';"
            )
            result = db.run(query)
            # Adjust depending on the return type of db.run (dict or list)
            if isinstance(result, dict) and "table_count" in result:
                table_count = result["table_count"]
            elif isinstance(result, list) and len(result) > 0 and "table_count" in result[0]:
                table_count = result[0]["table_count"]
            else:
                table_count = "an unknown number of"
            reply = f"There are {table_count} tables in the database."
        except Exception as e:
            reply = f"Failed to query the database: {str(e)}"
    else:
        # Convert session state messages to LangChain message objects
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
        
        # Get response from the local ChatOllama model
        response = model(messages)
        reply = response.content if hasattr(response, "content") else response[0].content

    # Append and display the assistant's reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
