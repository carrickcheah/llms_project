# agent001.py

import os
from loguru import logger
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import AgentType
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from utils.llm import initialize_openai, initialize_embeddings
from utils.maria import initialize_database
from utils.vector_database import qdrant_on_prem
from utils.tools import (
    limit_schema_size,
    load_sql_examples,
    create_sql_generation_prompt,
    initialize_sql_agent,
    answer_with_fallback,
)

# Main Function
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        relevant_tables = [table for table, _ in table_counts.most_common(20)]
        logger.info(f"Extracted tables from examples: {relevant_tables}")
        
        limited_toolkit = limit_schema_size(full_toolkit, max_tables=20, table_whitelist=relevant_tables)
        logger.success("SQLDatabaseToolkit with limited schema initialized successfully!")
        
        sql_agent = initialize_sql_agent(llm=llm, toolkit=limited_toolkit)
        schema_info = limited_toolkit.db.get_table_info()
        sql_generation_prompt = create_sql_generation_prompt(sql_examples, schema_info)
        sql_generation_chain = sql_generation_prompt | llm | StrOutputParser()
        logger.success("SQL generation chain initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to create SQLDatabaseToolkit or SQL agent: {e}")
        exit(1)

    local_prompt_template = """
    System: You are a friendly expert SQL data analyst. 

    Given an input question, determine if it’s database-related. If not, respond with: "Hi, I am SQL Agent, just answer database issue. Please ask something related to the database." 
    If database-related, create a syntactically correct {dialect} query to run, execute it, and return the answer with column names explored from the query.
    Use the following format for database questions:

    Question: "{input}"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result with column names identified (e.g., Customer_ID: value, Customer_Name: value)"
    Answer: "Final answer incorporating column names"
    Insight: "Optimize the Answer into a simple report, approximately 20 words"

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
        logger.success("LCEL RAG chain initialized successfully with local prompt.")
    except Exception as e:
        logger.error(f"Failed to initialize LCEL RAG chain: {str(e)}")
        raise

    print("Hi I am SQL Agent. How can I help you today? ")
    while True:
        try:
            user_query = input("User: ").strip()
            if not user_query:
                continue
            print(f"User: {user_query}")
            response = answer_with_fallback(user_query, qdrant_store, embeddings, db, sql_agent, sql_generation_chain, qa_chain, llm)
            print(f"Assistant: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"Assistant: Error: {str(e)}")

if __name__ == "__main__":
    main()