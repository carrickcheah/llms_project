import os
import time
from loguru import logger
import gradio as gr
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from utils.llm import initialize_openai, initialize_embeddings, initialize_open_deep
from utils.maria import initialize_database
from utils.vector_database import qdrant_on_prem
from utils.tools import (
    limit_schema_size,
    load_sql_examples,
    create_sql_generation_prompt,
    initialize_sql_agent,
    answer_with_fallback,
)

# Global initialization (to avoid reinitializing on each message)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm = initialize_openai(os.getenv("OPENAI_API_KEY"))
# llm = initialize_open_deep(os.getenv("DEEPSEEK_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

# Setup SQL Agent and chains (done once at startup)
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
    relevant_tables = [table for table, _ in table_counts.most_common(20)]
    logger.info(f"Extracted tables from examples: {relevant_tables}")
    
    limited_toolkit = limit_schema_size(full_toolkit, max_tables=20, table_whitelist=relevant_tables)
    sql_agent = initialize_sql_agent(llm=llm, toolkit=limited_toolkit)
    schema_info = limited_toolkit.db.get_table_info()
    sql_generation_prompt = create_sql_generation_prompt(sql_examples, schema_info)
    sql_generation_chain = sql_generation_prompt | llm | StrOutputParser()

    # Prompt template
    local_prompt_template = """
    System: You are a friendly expert SQL data analyst. 

    Given an input question, determine if it’s database-related. If not, respond with: "Hi, I am SQL Agent, just answer database issue. Please ask something related to the database." 
    If database-related, create a syntactically correct {dialect} query to run, execute it, and return the answer with column names explored from the query.
    Use the following format for database questions:

    Question: "{input}"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result with column names identified (e.g., Customer_ID: value, Customer_Name: value)"
    Answer: "Final answer incorporating column names"
    Insight: "Optimize the Answer into a simple report, 30 to 50 words"

    Only use the following tables:
    {table_info}

    Some examples of SQL queries that correspond to questions are:
    {few_shot_examples}
    """
    prompt = PromptTemplate(
        input_variables=["dialect", "input", "table_info", "few_shot_examples"],
        template=local_prompt_template
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        RunnableLambda(lambda x: {"question": x} if isinstance(x, str) else x)
        | {
            "context": qdrant_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
            "dialect": lambda x: db.dialect,
            "table_info": lambda x: db.get_table_info(),
            "few_shot_examples": lambda x: "",
            "input": lambda x: x["question"]
        }
        | prompt
        | llm
        | RunnableLambda(lambda x: x.content if hasattr(x, 'content') else (x.get('text', str(x)) if isinstance(x, dict) else str(x)))
    )
    logger.success("SQL Agent and chains initialized successfully!")
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise

# Streaming chat function for Gradio
def sql_agent_chat(message, history):
    """
    Streaming chat function for Gradio ChatInterface.
    Yields partial responses to simulate ChatGPT-like streaming behavior.
    """
    try:
        # Get the full response from the existing logic
        full_response = answer_with_fallback(message, qdrant_store, embeddings, db, sql_agent, sql_generation_chain, qa_chain, llm)
        
        # Simulate streaming by yielding chunks of the response
        current_response = ""
        for char in full_response:
            current_response += char
            time.sleep(0.02)  # Adjust delay for streaming speed
            yield current_response
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_message = f"Error: {str(e)}"
        for i in range(len(error_message)):
            time.sleep(0.02)
            yield error_message[:i + 1]

# Gradio Interface with ChatGPT-like UI
def main():
    demo = gr.ChatInterface(
        fn=sql_agent_chat,
        type="messages",
        title="Nex SQL Agent",
        description="Your friendly SQL data analyst. Ask me anything about the database!",
        examples=[
            "Show me the top 5 customers by sales",
            "Who is our top customer for Cycling Gloves in 2023?",
            "How many orders were placed this year?"
        ],
        theme="soft",  # A modern, clean theme similar to ChatGPT
        chatbot=gr.Chatbot(height=750, placeholder="Ask me a database question..."),
        textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7),
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
        save_history=True,  # Persistent chat history stored in browser
        css="""
            .chatbot {border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
            .message {font-family: 'Arial', sans-serif;}
            #component-0 {max-width: 2500px; margin: 0 auto;}  /* Center and limit width like ChatGPT */
            .textbox {max-width: 60px !important;}
        """
    )
    
    demo.launch()

if __name__ == "__main__":
    main()