# import os
# import json
# import ast
# from collections import Counter
# from loguru import logger
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from utils.llm import initialize_openai, initialize_embeddings
# from utils.maria import initialize_database
# from utils.vector_database import qdrant_on_prem
# from utils.tools import (
#     limit_schema_size, 
#     load_sql_examples, 
#     create_sql_generation_prompt, 
#     initialize_sql_agent, 
#     is_database_question, 
#     test_embedding_retrieval, 
# )

# # Utility Functions

# def clean_sql_query(sql_query):
#     """Remove markdown tags from the SQL query if present."""
#     lines = sql_query.split('\n')
#     if lines[0].strip() == '```sql' and lines[-1].strip() == '```':
#         return '\n'.join(lines[1:-1])
#     return sql_query

# def format_sql_result(result):
#     """Format the SQL result for better readability."""
#     try:
#         result_list = ast.literal_eval(result)
#         if result_list:
#             row = result_list[0]
#             if len(row) == 1:
#                 return f"The top customer is {row[0]}"
#             elif len(row) == 3:
#                 return f"The top customer is {row[1]} with total revenue of {row[2]}"
#             else:
#                 return str(result_list)
#         else:
#             return "No result found"
#     except:
#         return result

# # Main Function

# def main():
#     """Initialize components and run the chat loop."""
#     # Suppress HuggingFace tokenizers parallelism warning
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"

#     # Initialize components
#     llm = initialize_openai(os.getenv("OPENAI_API_KEY"))
#     embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
#     db = initialize_database(os.getenv("DATABASE_URI"))
#     qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
#     sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

#     # Initialize SQL toolkit and agent
#     try:
#         full_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
#         table_mentions = []
#         for example in sql_examples:
#             sql = example.get('sql_query', '').lower()
#             for word in ['from', 'join']:
#                 if word in sql:
#                     parts = sql.split(word)[1].strip().split()
#                     if parts:
#                         table = parts[0].strip(';,() ')
#                         if table:
#                             table_mentions.append(table)
        
#         table_counts = Counter(table_mentions)
#         relevant_tables = [table for table, _ in table_counts.most_common(15)]
        
#         logger.info(f"Extracted tables from examples: {relevant_tables}")
        
#         limited_toolkit = limit_schema_size(
#             full_toolkit, 
#             max_tables=15,
#             table_whitelist=relevant_tables
#         )
        
#         logger.success("SQLDatabaseToolkit with limited schema initialized successfully!")
#         sql_agent = initialize_sql_agent(llm=llm, toolkit=limited_toolkit)
        
#         schema_info = limited_toolkit.db.get_table_info()
#         sql_generation_prompt = create_sql_generation_prompt(sql_examples, schema_info)
#         sql_generation_chain = sql_generation_prompt | llm | StrOutputParser()
#         logger.success("SQL generation chain initialized successfully!")
        
#     except Exception as e:
#         logger.error(f"Failed to create SQLDatabaseToolkit or SQL agent: {e}")
#         exit(1)

#     # Define the local prompt template
#     local_prompt_template = """
#     Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
#     Use the following format:

#     Question: "{input}"
#     SQLQuery: "SQL Query to run"
#     SQLResult: "Result of the SQLQuery"
#     Answer: "Final answer here"

#     Only use the following tables:
#     {table_info}

#     Some examples of SQL queries that correspond to questions are:
#     {few_shot_examples}
#     """

#     # Initialize RAG with local prompt
#     try:
#         prompt = PromptTemplate(
#             input_variables=["dialect", "input", "table_info", "few_shot_examples"],
#             template=local_prompt_template
#         )
#         logger.info("Local prompt template created successfully.")
        
#         def format_docs(docs):
#             return "\n\n".join(doc.page_content for doc in docs)

#         qa_chain = (
#             {
#                 "context": qdrant_store.as_retriever() | format_docs,
#                 "question": RunnablePassthrough(),
#                 "dialect": lambda x: db.dialect,
#                 "table_info": lambda x: db.get_table_info(),
#                 "few_shot_examples": lambda x: "",
#                 "input": lambda x: x["question"]
#             }
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
#         logger.success("LCEL RAG chain initialized successfully with local prompt.")
#     except Exception as e:
#         logger.error(f"Failed to initialize LCEL RAG chain: {str(e)}")
#         raise

#     # Chat loop
#     print("Hi I am SQL Agent. How can i help you today? ")
#     while True:
#         try:
#             user_query = input("User: ").strip()
#             if not user_query:
#                 continue
#             print(f"User: {user_query}")
#             response = answer_with_fallback(user_query, qdrant_store, embeddings, db, sql_agent, sql_generation_chain, qa_chain, llm)
#             print(f"Assistant: {response}")
#         except KeyboardInterrupt:
#             print("\nExiting chat...")
#             break
#         except Exception as e:
#             logger.error(f"Error in chat loop: {str(e)}")
#             print(f"Assistant: Error: {str(e)}")

# # Answer Function with Fallback

# def answer_with_fallback(question, qdrant_store, embeddings, db, sql_agent, sql_generation_chain, qa_chain, llm):
#     """Handle user questions with fallback logic for SQL and RAG responses."""
#     try:
#         logger.info(f"Running diagnostic test for question: {question}")
#         retrieved_docs = test_embedding_retrieval(question, qdrant_store, embeddings)
        
#         if is_database_question(question):
#             logger.info(f"Database question detected: {question}")
            
#             if retrieved_docs:
#                 try:
#                     sql_examples = []
#                     for doc in retrieved_docs:
#                         try:
#                             example_q = doc.page_content
#                             example_sql = doc.metadata.get("sql_query", "") if isinstance(doc.metadata, dict) else str(doc.metadata)
#                             if example_q and example_sql:
#                                 sql_examples.append({"question": example_q, "sql": example_sql})
#                         except Exception as e:
#                             logger.warning(f"Error parsing retrieved document: {str(e)}")
#                             continue
                    
#                     if sql_examples:
#                         logger.info(f"Creating SQL generation prompt with {len(sql_examples)} retrieved examples")
#                         examples_text = "\n\n".join([f"Similar Question: {ex['question']}\nCorresponding SQL: {ex['sql']}" for ex in sql_examples])
                        
#                         retrieval_prompt = ChatPromptTemplate.from_template("""
#                         You are an expert in converting natural language questions to SQL queries.
#                         I want to answer this question: {question}
#                         Here are similar questions with their corresponding SQL queries:
#                         {examples}
#                         Based on these examples, generate a SQL query that will answer my question.
#                         Only return the SQL query without any explanation or comments.
#                         """)
                        
#                         logger.info("Generating SQL using retrieved examples...")
#                         retrieval_chain = retrieval_prompt | llm | StrOutputParser()
#                         sql_query = retrieval_chain.invoke({"question": question, "examples": examples_text})
                        
#                         # Clean the SQL query
#                         sql_query = clean_sql_query(sql_query)
#                         logger.info(f"Generated SQL using retrieved examples: {sql_query}")
                        
#                         try:
#                             logger.info("Executing SQL query from retrieved examples...")
#                             result = db.run(sql_query)
#                             formatted_result = format_sql_result(result)
#                             return f"SQL Database Answer (using retrieved examples):\nQuery: {sql_query}\n\n{formatted_result}"
#                         except Exception as sql_exec_error:
#                             logger.warning(f"Error executing SQL from retrieved examples: {str(sql_exec_error)}")
#                     else:
#                         logger.warning("No usable SQL examples found in retrieved documents")
#                 except Exception as retrieval_error:
#                     logger.warning(f"Error using retrieved examples: {str(retrieval_error)}")
            
#             # Fallback to file examples
#             try:
#                 logger.info(f"Generating SQL query using file examples for: {question}")
#                 sql_query = sql_generation_chain.invoke({"question": question})
#                 sql_query = clean_sql_query(sql_query)  # Clean the query
#                 logger.info(f"Generated SQL query from file examples: {sql_query}")
                
#                 try:
#                     result = db.run(sql_query)
#                     formatted_result = format_sql_result(result)
#                     return f"SQL Database Answer (using file examples):\nQuery: {sql_query}\n\n{formatted_result}"
#                 except Exception as sql_exec_error:
#                     logger.warning(f"Error executing generated SQL: {str(sql_exec_error)}")
                    
#                     # Fallback to agent
#                     logger.info(f"Falling back to SQL agent for: {question}")
#                     agent_response = sql_agent.invoke({"input": question})
#                     if isinstance(agent_response, dict) and "output" in agent_response:
#                         return f"SQL Database Answer (via agent): {agent_response['output']}"
#                     return f"SQL Database Answer (via agent): {agent_response}"
            
#             except Exception as sql_gen_error:
#                 logger.error(f"SQL generation failed: {str(sql_gen_error)}")
#                 return "I wasn't able to find relevant data to answer your question with the available tools."
#         else:
#             logger.info(f"Non-database question detected. Trying to answer with RAG: {question}")
#             response = qa_chain.invoke(question)
#             if "I don't know" in response:
#                 logger.info(f"RAG couldn't answer. Falling back to general response.")
#                 return f"I don't have specific information about {question}. You might want to rephrase your query or ask about shipping methods, order data, or customer information that's available in the database."
#             return f"Knowledge Base Answer: {response}"
            
#     except Exception as e:
#         logger.error(f"Error answering question '{question}': {str(e)}")
#         return f"Error: Could not process the question due to: {str(e)}"