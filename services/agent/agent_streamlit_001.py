# this streamlit agent with meta data
#

import os
import streamlit as st
from loguru import logger
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from utils.llm import initialize_openai, initialize_embeddings
from utils.maria import initialize_database
from utils.vector_database import qdrant_on_prem
from utils.tools import (
    limit_schema_size,
    load_sql_examples,
    create_sql_generation_prompt,
    initialize_sql_agent,
    answer_with_fallback,
    generate_report
)

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize components at module level to avoid PyTorch/Streamlit conflicts
llm = initialize_openai(os.getenv("OPENAI_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

try:
    full_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    table_mentions = []
    for example in sql_examples:
        sql = example.get('sql_query', '').lower()
        for word in ['from', 'join']:
            if word in sql:
                parts = sql.split(word)[1].strip().split()
                if parts:
                    table = parts[0].strip(';,() ')
                    if table:
                        table_mentions.append(table)
    from collections import Counter
    table_counts = Counter(table_mentions)
    relevant_tables = [table for table, _ in table_counts.most_common(15)]
    logger.info(f"Extracted tables from examples: {relevant_tables}")
    
    limited_toolkit = limit_schema_size(full_toolkit, max_tables=15, table_whitelist=relevant_tables)
    logger.success("SQLDatabaseToolkit with limited schema initialized successfully!")
    sql_agent = initialize_sql_agent(llm=llm, toolkit=limited_toolkit)
    
    schema_info = limited_toolkit.db.get_table_info()
    sql_generation_prompt = create_sql_generation_prompt(sql_examples, schema_info)
    sql_generation_chain = sql_generation_prompt | llm | StrOutputParser()
    logger.success("SQL generation chain initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

local_prompt_template = """
Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:
Question: "{input}"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the following tables:
{table_info}
Some examples of SQL queries that correspond to questions are:
{few_shot_examples}
"""
try:
    prompt = PromptTemplate(
        input_variables=["dialect", "input", "table_info", "few_shot_examples"],
        template=local_prompt_template
    )
    logger.info("Local prompt template created successfully.")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": qdrant_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
            "dialect": lambda x: db.dialect,
            "table_info": lambda x: db.get_table_info(),
            "few_shot_examples": lambda x: "",
            "input": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.success("LCEL RAG chain initialized successfully with local prompt.")
except Exception as e:
    logger.error(f"Failed to initialize RAG chain: {str(e)}")
    raise

# Streamlit app with ChatGPT-like UI
def main():
    st.title("SQL Agent Chat")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I’m your SQL Agent. Ask me anything about the database, and I’ll provide answers and reports!"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input at the bottom
    user_query = st.chat_input("Type your question here...")

    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process query and generate response
        with st.spinner("Thinking..."):
            try:
                response = answer_with_fallback(user_query, qdrant_store, embeddings, db, sql_agent, sql_generation_chain, qa_chain, llm)
                full_response = response
                
                # Generate report if SQL-based answer
                if "SQL Database Answer" in response:
                    lines = response.split('\n')
                    sql_query = lines[1].replace("Query: ", "").strip()
                    result = lines[3].strip()
                    report = generate_report(llm, user_query, sql_query, result)
                    full_response = f"{response}\n\n**Report:**\n{report}"
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with st.chat_message("assistant"):
                    st.markdown(full_response)
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                error_msg = f"Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)

if __name__ == "__main__":
    main()