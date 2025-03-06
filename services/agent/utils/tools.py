########################################################################################
# Contents of table (Tools)                                                            #
# 1. limit_schema_size: Restricts database schema to prevent token overflows.          #
# 2. load_sql_examples: Loads SQL examples from a file.                                #
# 3. create_sql_generation_prompt: Creates NL-to-SQL conversion template.              #
# 4. initialize_sql_agent: Sets up SQL conversion/execution agent.                     #
# 5. is_database_question: Checks if query needs SQL.                                  #
# 6. test_embedding_retrieval: Tests vector search functionality.                      #
# 7. answer_with_fallback: Multi-method question answering with fallbacks.             #
# 8. clean_sql_query: Removes markdown tags from SQL queries.                          #
########################################################################################

import json
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import AgentType
from langgraph.func import task  # Import for @task decorator

########################################################################################
##     1.   limit_schema_size: Restricts database schema to prevent token overflows   ##
########################################################################################

def limit_schema_size(toolkit, max_tables=20, table_blacklist=None, table_whitelist=None):
    """Limit the schema size to prevent exceeding token limits."""
    db = toolkit.db
    
    all_tables = db.get_usable_table_names()
    logger.info(f"Total tables in database: {len(all_tables)}")
    
    tables_to_include = all_tables
    
    if table_whitelist:
        tables_to_include = [t for t in all_tables if t in table_whitelist]
        logger.info(f"Using whitelist. Including only {len(tables_to_include)} tables.")
    elif table_blacklist:
        tables_to_include = [t for t in all_tables if t not in table_blacklist]
        logger.info(f"Using blacklist. Including {len(tables_to_include)} tables.")
    
    if len(tables_to_include) > max_tables and max_tables > 0:
        tables_to_include = tables_to_include[:max_tables]
        logger.info(f"Limited to {max_tables} tables.")
    
    limited_db = SQLDatabase(
        engine=db._engine,
        schema=db._schema,
        metadata=db._metadata,
        include_tables=tables_to_include,
        sample_rows_in_table_info=2
    )
    
    limited_toolkit = SQLDatabaseToolkit(db=limited_db, llm=toolkit.llm)
    
    logger.info(f"Limited schema to tables: {', '.join(tables_to_include)}")
    
    return limited_toolkit

########################################################################################
##                2. load_sql_examples: Loads SQL examples from a file.               ##
########################################################################################

def load_sql_examples(examples_file_path):
    """Load SQL examples from a JSONL file."""
    logger.info(f"Loading SQL examples from {examples_file_path}")
    examples = []
    
    try:
        with open(examples_file_path, 'r') as file:
            for line in file:
                if line.strip():
                    examples.append(json.loads(line))
        
        logger.success(f"Loaded {len(examples)} SQL examples")
        return examples
    except Exception as e:
        logger.error(f"Failed to load SQL examples: {str(e)}")
        return []

########################################################################################
##        3.  create_sql_generation_prompt: Creates NL-to-SQL conversion template     ##
########################################################################################

def create_sql_generation_prompt(examples, schema_info):
    """Create a prompt for SQL generation using examples."""
    few_shot_examples = ""
    for idx, example in enumerate(examples[:5]):
        question = example.get('page_content', '')
        sql = example.get('sql_query', '')
        few_shot_examples += f"Example {idx+1}:\nQuestion: {question}\nSQL: {sql}\n\n"
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert in converting natural language questions to SQL queries.
        
        Database Schema:
        {schema}
        
        Here are some examples of natural language questions and their corresponding SQL queries:
        
        {examples}
        
        Now, convert the following question to a valid SQL query:
        Question: {question}
        
        Respond with ONLY the SQL query and nothing else. Do not include ```sql or ``` tags."""
    )
    
    return prompt.partial(schema=schema_info, examples=few_shot_examples)

########################################################################################
##          4. initialize_sql_agent: Sets up SQL conversion/execution agent            #
########################################################################################

def initialize_sql_agent(llm, toolkit):
    """Initialize an SQL agent for natural language to SQL conversion and execution."""
    try:
        logger.info("Initializing SQL agent...")
        sql_agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        logger.success("SQL agent initialized successfully!")
        return sql_agent
    except Exception as e:
        logger.error(f"Failed to initialize SQL agent: {str(e)}")
        raise ConnectionError(f"Failed to initialize SQL agent: {str(e)}")

########################################################################################
##              5. is_database_question: Checks if query needs SQL                    ##
########################################################################################

def is_database_question(question):
    """Determine if a question is likely a database query that requires SQL execution."""
    db_keywords = [
        'shipping', 'method', 'revenue', 'order', 'customer', 'value', 
        'month', 'age', 'decade', 'express', 'standard', 'premium', 
        'count', 'total', 'average', 'highest', 'lowest', 'most', 'least',
        'popular', 'common', 'price', 'cost', 'quantity', 'frequency',
        'how many', 'which', 'what is', 'what are', 'percentage','purchase',
        'receipt'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in db_keywords)

########################################################################################
##            6. test_embedding_retrieval: Tests vector search functionality           #
########################################################################################

def test_embedding_retrieval(query, qdrant_store, embeddings):
    """Test if embeddings model can successfully retrieve documents from Qdrant."""
    logger.info("=" * 50)
    logger.info(f"DIAGNOSTIC TEST: TESTING EMBEDDING RETRIEVAL FOR: '{query}'")
    
    try:
        logger.info("Generating embedding vector for query...")
        embedding_vector = embeddings.embed_query(query)
        embedding_dims = len(embedding_vector)
        logger.info(f"SUCCESS: Generated embedding vector with {embedding_dims} dimensions")
    except Exception as e:
        logger.error(f"FAILURE: Could not generate embedding: {str(e)}")
        return None
    
    retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})
    
    try:
        logger.info("Retrieving documents from Qdrant...")
        retrieved_docs = retriever.invoke(query)
        
        if retrieved_docs:
            logger.success(f"SUCCESS: Retrieved {len(retrieved_docs)} documents from Qdrant")
            
            for i, doc in enumerate(retrieved_docs):
                logger.info(f"Document #{i+1}:")
                
                if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                    score = doc.metadata['score']
                    logger.info(f"  Score: {score}")
                
                logger.info(f"  Raw content (first 100 chars): {doc.page_content[:100]}...")
                
                user_query = doc.page_content
                sql_query = doc.metadata.get("sql_query", "") if isinstance(doc.metadata, dict) else str(doc.metadata)
                
                logger.success(f"  QUESTION: '{user_query}'")
                logger.success(f"  SQL: '{sql_query[:100]}...'")
            
            return retrieved_docs
        else:
            logger.warning("NO DOCUMENTS RETRIEVED! Check your embedding model and Qdrant collection.")
            return None
    
    except Exception as e:
        logger.error(f"ERROR during retrieval: {str(e)}")
        return None
    
    finally:
        logger.info("=" * 50)

########################################################################################
##       7. answer_with_fallback: Multi-method question answering with fallbacks       #
########################################################################################

def answer_with_fallback(question, qdrant_store, embeddings, db, sql_agent, sql_generation_chain, qa_chain, llm):
    """Answer a question using multiple methods with fallbacks."""
    try:
        simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if question.strip().lower() in simple_greetings:
            return "Hi, I am SQL Agent, just answer database issue. Please ask something related to the database."

        logger.info(f"Running diagnostic test for question: {question}")
        retrieved_docs = test_embedding_retrieval(question, qdrant_store, embeddings)
        
        if is_database_question(question):
            logger.info(f"Database question detected: {question}")
            if retrieved_docs:
                sql_examples = []
                exact_match_sql = None
                for doc in retrieved_docs:
                    try:
                        example_q = doc.page_content
                        example_sql = doc.metadata.get("sql_query", "") if isinstance(doc.metadata, dict) else str(doc.metadata)
                        if example_q and example_sql:
                            sql_examples.append({"question": example_q, "sql": example_sql})
                            if example_q.strip() == question.strip():
                                exact_match_sql = example_sql
                    except Exception as e:
                        logger.warning(f"Error parsing retrieved document: {str(e)}")
                        continue
                
                # Dynamic prompt with executed result
                dynamic_prompt = ChatPromptTemplate.from_template("""
                Given an input question and its executed SQL result, return the answer with column names explored from the query.
                Use the following format:

                Question: "{question}"
                SQLQuery: "{sql_query}"
                SQLResult: "Result with column names identified (e.g., Customer_ID: value, Customer_Name: value)"
                Answer: "Final answer incorporating column names"
                Insight: "Optimize the Answer into a simple report, approximately 20 words"

                Question: "{question}"
                SQLQuery: "{sql_query}"
                SQLResult: "{result}"
                """)

                if exact_match_sql:
                    sql_query = clean_sql_query(exact_match_sql)
                    logger.info(f"Using exact match retrieved SQL: {sql_query}")
                    try:
                        result = db.run(sql_query)
                        response_chain = dynamic_prompt | llm | StrOutputParser()
                        response = response_chain.invoke({"question": question, "sql_query": sql_query, "result": result})
                        return response
                    except Exception as sql_exec_error:
                        logger.warning(f"Error executing exact match SQL: {str(sql_exec_error)}")
                
                if sql_examples:
                    logger.info(f"Creating SQL generation prompt with {len(sql_examples)} retrieved examples")
                    examples_text = "\n\n".join([f"Similar Question: {ex['question']}\nCorresponding SQL: {ex['sql']}" for ex in sql_examples])
                    retrieval_prompt = ChatPromptTemplate.from_template("""
                    You are an expert in converting natural language questions to SQL queries.
                    I want to answer this question: {question}
                    Here are similar questions with their corresponding SQL queries:
                    {examples}
                    Based on these examples, generate a SQL query that will answer my question.
                    Only return the SQL query without any explanation or comments.
                    """)
                    logger.info("Generating SQL using retrieved examples...")
                    retrieval_chain = retrieval_prompt | llm | StrOutputParser()
                    sql_query = retrieval_chain.invoke({"question": question, "examples": examples_text})
                    sql_query = clean_sql_query(sql_query)
                    logger.info(f"Generated SQL using retrieved examples: {sql_query}")
                    try:
                        result = db.run(sql_query)
                        response_chain = dynamic_prompt | llm | StrOutputParser()
                        response = response_chain.invoke({"question": question, "sql_query": sql_query, "result": result})
                        return response
                    except Exception as sql_exec_error:
                        logger.warning(f"Error executing SQL from retrieved examples: {str(sql_exec_error)}")
            
            try:
                logger.info(f"Generating SQL query using file examples for: {question}")
                sql_query = sql_generation_chain.invoke({"question": question})
                sql_query = clean_sql_query(sql_query)
                logger.info(f"Generated SQL query from file examples: {sql_query}")
                try:
                    result = db.run(sql_query)
                    response_chain = dynamic_prompt | llm | StrOutputParser()
                    response = response_chain.invoke({"question": question, "sql_query": sql_query, "result": result})
                    return response
                except Exception as sql_exec_error:
                    logger.warning(f"Error executing generated SQL: {str(sql_exec_error)}")
                    logger.info(f"Falling back to SQL agent for: {question}")
                    agent_response = sql_agent.invoke({"input": question})
                    if isinstance(agent_response, dict) and "output" in agent_response:
                        # Wrap agent response with dynamic prompt
                        response_chain = dynamic_prompt | llm | StrOutputParser()
                        response = response_chain.invoke({"question": question, "sql_query": "Unknown (agent-generated)", "result": agent_response['output']})
                        return response
                    return str(agent_response)
            except Exception as sql_gen_error:
                logger.error(f"SQL generation failed: {str(sql_gen_error)}")
                return "I wasn't able to find relevant data to answer your question with the available tools."
        else:
            logger.info(f"Non-database question detected. Trying to answer with RAG: {question}")
            return qa_chain.invoke(question)
    except Exception as e:
        logger.error(f"Error answering question '{question}': {str(e)}")
        return f"Error: Could not process the question due to: {str(e)}"

########################################################################################
##                                9. clean_sql_query                                  ##
########################################################################################

def clean_sql_query(sql_query):
    """Remove markdown tags from the SQL query if present."""
    lines = sql_query.split('\n')
    if lines[0].strip() == '```sql' and lines[-1].strip() == '```':
        return '\n'.join(lines[1:-1])
    return sql_query
