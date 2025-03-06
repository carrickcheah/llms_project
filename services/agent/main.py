# main.py ( always keep this # comment )
import os
from loguru import logger
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver
from hashlib import md5

# Custom utility imports
from utils.llm import initialize_openai, initialize_embeddings
from utils.maria import initialize_database
from utils.vector_database import qdrant_on_prem
from utils.task import process_database_question
from utils.tools import (
    limit_schema_size,
    load_sql_examples,
    create_sql_generation_prompt,
    initialize_sql_agent,
    is_database_question
)

# Global initialization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm = initialize_openai(os.getenv("OPENAI_API_KEY"))
embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
db = initialize_database(os.getenv("DATABASE_URI"))
qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

# Global cache for table detection
TABLE_CACHE = {}

# Prompt template with Qdrant retrieval
local_prompt_template = """
System: You are a friendly expert SQL data analyst.

Given an input question, create a syntactically correct {dialect} query to run against the database.
Use the following format for the response:

Question: "{input}"
SQLQuery: "SQL Query to run"
SQLResult: "Result with column names identified (e.g., Customer_ID: value, Customer_Name: value)"
Answer: "Final answer incorporating column names"
Insight: "Optimize the Answer into a simple report, approximately 20 words"

Only use the following tables:
{table_info}

Relevant context from vector search (past queries or schema info):
{qdrant_context}

Examples of SQL queries for similar questions:
{few_shot_examples}
"""
prompt = PromptTemplate(
    input_variables=["dialect", "input", "table_info", "qdrant_context", "few_shot_examples"],
    template=local_prompt_template
)

# QA chain for non-database questions
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
logger.success("LCEL RAG chain initialized successfully with local prompt!")

# Polite response for non-database questions
def generate_polite_response(question: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
    You are a friendly SQL Agent. Your role is to assist users with database-related questions.
    If a user asks something unrelated to the database, respond politely and guide them to ask a database-related question.

    User's input: "{question}"
    Your response:
    """)
    response_chain = prompt | llm | StrOutputParser()
    return response_chain.invoke({"question": question})

# Dynamic table detection with LLM
def detect_relevant_tables(question: str, db: SQLDatabase, llm) -> list:
    cache_key = md5(question.encode()).hexdigest()
    if cache_key in TABLE_CACHE:
        logger.info(f"Using cached table detection for question: {question}")
        return TABLE_CACHE[cache_key]

    table_names = db.get_usable_table_names()
    prompt = ChatPromptTemplate.from_template("""
    You are an expert SQL analyst. Given a database question and a list of available tables,
    identify which tables are most likely needed to answer the question.
    Return only the table names as a comma-separated list (e.g., "sales, customers").
    If unsure, return an empty string.

    Question: "{question}"
    Available tables: {table_names}
    Table names:
    """)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question, "table_names": ", ".join(table_names)})
    
    detected_tables = [t.strip() for t in response.split(",") if t.strip() in table_names]
    if not detected_tables:
        logger.warning(f"No tables detected for question: {question}. Using fallback.")
        detected_tables = table_names[:3] if len(table_names) >= 3 else table_names
    
    TABLE_CACHE[cache_key] = detected_tables
    logger.info(f"Detected tables for '{question}': {detected_tables}")
    return detected_tables

# Search Qdrant for relevant context
def search_qdrant_context(question: str, qdrant_store, embeddings, top_k=3) -> str:
    question_embedding = embeddings.embed_query(question)
    search_results = qdrant_store.search(question_embedding, limit=top_k)
    context = "\n".join([f"- {result.payload.get('content', 'No content')}" for result in search_results])
    logger.info(f"Qdrant context retrieved for '{question}':\n{context}")
    return context if context else "No relevant context found in Qdrant."

# Main workflow with corrected sql_generation_chain
@entrypoint(checkpointer=MemorySaver())
def sql_agent_workflow(inputs: dict) -> dict:
    question = inputs.get("question", "").strip()
    if not question:
        return {"response": "Please provide a question."}

    if is_database_question(question):
        logger.info(f"Database question detected: {question}")
        
        # Define the SQL generation chain dynamically
        sql_generation_chain = (
            RunnableLambda(lambda x: {"question": x}) |
            RunnablePassthrough.assign(qdrant_context=lambda x: search_qdrant_context(x["question"], qdrant_store, embeddings)) |
            RunnablePassthrough.assign(table_info=lambda x: db.get_table_info(detect_relevant_tables(x["question"], db, llm))) |
            RunnablePassthrough.assign(prompt=lambda x: prompt.format(
                dialect=db.dialect,
                input=x["question"],
                table_info=x["table_info"],
                qdrant_context=x["qdrant_context"],
                few_shot_examples="\n".join([f"- Q: {ex['question']}\n  SQL: {ex['sql_query']}" for ex in sql_examples[:3]])
            )) |
            llm |
            StrOutputParser()
        )
        
        # Generate and execute SQL
        response = process_database_question(question, sql_generation_chain, db, llm).result()
    else:
        logger.info(f"Non-database question detected: {question}")
        response = generate_polite_response(question)

    return {"response": response}

# Interactive execution
def run_interactive():
    config = {"configurable": {"thread_id": "sql_agent_thread"}}
    print("Hi I am SQL Agent. How can I help you today?")
    while True:
        try:
            user_query = input("User: ").strip()
            if not user_query:
                continue
            print(f"User: {user_query}")

            result = sql_agent_workflow.invoke({"question": user_query}, config=config)
            print(f"Assistant: {result['response']}")

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"Assistant: Error: {str(e)}")

if __name__ == "__main__":
    run_interactive()