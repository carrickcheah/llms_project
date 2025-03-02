from typing import List
import os
import json
from loguru import logger
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from llm import initialize_llm
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def use_existing_qdrant(
    embedding: Embeddings,
    url: str = "http://localhost:6333",
    collection_name: str = "query_embeddings",
    prefer_grpc: bool = True
) -> QdrantVectorStore:
    """Retrieve or initialize an existing Qdrant vector store."""
    if not embedding:
        logger.error("Embedding model is not provided.")
        raise ValueError("An embedding model must be provided.")

    try:
        logger.info(f"Connecting to existing Qdrant vector store at {url} with collection '{collection_name}' "
                    f"(prefer_grpc={prefer_grpc})")
        
        client = QdrantClient(url=url, prefer_grpc=prefer_grpc)
        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding
        )
        
        logger.success(f"Successfully connected to Qdrant vector store '{collection_name}'.")
        return qdrant_store
    
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant vector store: {str(e)}")
        raise ConnectionError(f"Failed to connect to Qdrant vector store: {str(e)}")

def limit_schema_size(toolkit, max_tables=10, table_blacklist=None, table_whitelist=None):
    """Limit the schema size to prevent exceeding token limits."""
    # Get the database instance from the toolkit
    db = toolkit.db
    
    # Get all table names
    all_tables = db.get_usable_table_names()
    logger.info(f"Total tables in database: {len(all_tables)}")
    
    # Determine tables to include
    tables_to_include = all_tables
    
    if table_whitelist:
        # If whitelist is provided, only include those tables
        tables_to_include = [t for t in all_tables if t in table_whitelist]
        logger.info(f"Using whitelist. Including only {len(tables_to_include)} tables.")
    elif table_blacklist:
        # If blacklist is provided, exclude those tables
        tables_to_include = [t for t in all_tables if t not in table_blacklist]
        logger.info(f"Using blacklist. Including {len(tables_to_include)} tables.")
    
    # Limit to max_tables
    if len(tables_to_include) > max_tables and max_tables > 0:
        tables_to_include = tables_to_include[:max_tables]
        logger.info(f"Limited to {max_tables} tables.")
    
    # Create a new database instance with limited tables
    limited_db = SQLDatabase(
        engine=db._engine,
        schema=db._schema,
        metadata=db._metadata,
        include_tables=tables_to_include,
        sample_rows_in_table_info=2  # Reduce sample rows to save tokens
    )
    
    # Create a new toolkit with the limited database
    limited_toolkit = SQLDatabaseToolkit(db=limited_db, llm=toolkit.llm)
    
    # Log the tables included
    logger.info(f"Limited schema to tables: {', '.join(tables_to_include)}")
    
    return limited_toolkit

def load_sql_examples(examples_file_path):
    """Load SQL examples from a JSONL file."""
    logger.info(f"Loading SQL examples from {examples_file_path}")
    examples = []
    
    try:
        with open(examples_file_path, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    examples.append(json.loads(line))
        
        logger.success(f"Loaded {len(examples)} SQL examples")
        return examples
    except Exception as e:
        logger.error(f"Failed to load SQL examples: {str(e)}")
        return []

def create_sql_generation_prompt(examples, schema_info):
    """Create a prompt for SQL generation using examples."""
    # Build the few-shot examples text
    few_shot_examples = ""
    for idx, example in enumerate(examples[:5]):  # Use first 5 examples to save tokens
        question = example.get('sentence1_column', '')
        sql = example.get('sentence2_column', '')
        few_shot_examples += f"Example {idx+1}:\nQuestion: {question}\nSQL: {sql}\n\n"
    
    # Create the prompt template
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
    
    # Return the populated prompt
    return prompt.partial(schema=schema_info, examples=few_shot_examples)

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

# Function to determine if a question is a database query
def is_database_question(question):
    """Determine if a question is likely a database query that requires SQL execution."""
    db_keywords = [
        'shipping', 'method', 'revenue', 'order', 'customer', 'value', 
        'month', 'age', 'decade', 'express', 'standard', 'premium', 
        'count', 'total', 'average', 'highest', 'lowest', 'most', 'least',
        'popular', 'common', 'price', 'cost', 'quantity', 'frequency',
        'how many', 'which', 'what is', 'what are', 'percentage'
    ]
    
    question_lower = question.lower()
    
    # Check if any database keywords are present in the question
    return any(keyword in question_lower for keyword in db_keywords)

# NEW FUNCTION: Test embeddings and Qdrant retrieval
def test_embedding_retrieval(query, qdrant_store, embeddings):
    """Test if embeddings model can successfully retrieve documents from Qdrant."""
    logger.info("=" * 50)
    logger.info(f"DIAGNOSTIC TEST: TESTING EMBEDDING RETRIEVAL FOR: '{query}'")
    
    # First, log the embedding process
    try:
        logger.info("Generating embedding vector for query...")
        embedding_vector = embeddings.embed_query(query)
        embedding_dims = len(embedding_vector)
        logger.info(f"SUCCESS: Generated embedding vector with {embedding_dims} dimensions")
    except Exception as e:
        logger.error(f"FAILURE: Could not generate embedding: {str(e)}")
        return None
    
    # Set up retriever with search parameters
    retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})
    
    # Get relevant documents from Qdrant
    try:
        logger.info("Retrieving documents from Qdrant...")
        retrieved_docs = retriever.get_relevant_documents(query)
        
        if retrieved_docs:
            logger.success(f"SUCCESS: Retrieved {len(retrieved_docs)} documents from Qdrant")
            
            # Log each retrieved document
            for i, doc in enumerate(retrieved_docs):
                logger.info(f"Document #{i+1}:")
                
                # Try to extract score if available
                score = None
                if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                    score = doc.metadata['score']
                    logger.info(f"  Score: {score}")
                
                # Log raw content sample
                logger.info(f"  Raw content (first 100 chars): {doc.page_content[:100]}...")
                
                # Try to parse different possible JSON formats
                try:
                    doc_data = json.loads(doc.page_content)
                    
                    # Try format 1 (sentence1_column/sentence2_column)
                    if "sentence1_column" in doc_data and "sentence2_column" in doc_data:
                        user_query = doc_data.get("sentence1_column", "")
                        sql_query = doc_data.get("sentence2_column", "")
                        logger.success(f"  FOUND FORMAT 1 - Question: '{user_query}'")
                        logger.success(f"  SQL: '{sql_query[:100]}...'")
                    
                    # Try format 2 (user_query/generated_sql)
                    elif "user_query" in doc_data and "generated_sql" in doc_data:
                        user_query = doc_data.get("user_query", "")
                        sql_query = doc_data.get("generated_sql", "")
                        logger.success(f"  FOUND FORMAT 2 - Question: '{user_query}'")
                        logger.success(f"  SQL: '{sql_query[:100]}...'")
                    
                    # Unknown format - log keys
                    else:
                        keys = list(doc_data.keys())
                        logger.warning(f"  UNKNOWN FORMAT - Keys in document: {keys}")
                        
                        # Try to find any field that might contain a question
                        for key in keys:
                            value = doc_data.get(key, "")
                            if isinstance(value, str) and '?' in value:
                                logger.info(f"  Possible question field '{key}': '{value}'")
                
                except json.JSONDecodeError:
                    logger.warning(f"  NOT JSON FORMAT - Document appears to be plain text")
                except Exception as e:
                    logger.error(f"  ERROR parsing document: {str(e)}")
            
            return retrieved_docs
        else:
            logger.warning("NO DOCUMENTS RETRIEVED! Check your embedding model and Qdrant collection.")
            return None
    
    except Exception as e:
        logger.error(f"ERROR during retrieval: {str(e)}")
        return None
    
    finally:
        logger.info("=" * 50)

if __name__ == "__main__":
    # Suppress LangSmith warning if not using LangSmith
    warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

    # Initialize embeddings for the existing vector store
    try:
        embeddings = HuggingFaceEmbeddings(model_name="carrick113/autotrain-wsucv-dqrgc")
        logger.info("HuggingFace embeddings initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise

    # Connect to the existing Qdrant vector store
    try:
        qdrant_store = use_existing_qdrant(
            embedding=embeddings,
            url="http://localhost:6333",
            collection_name="query_embeddings",
            prefer_grpc=True
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant vector store: {str(e)}")
        raise

    # Initialize the database
    try:
        db = SQLDatabase.from_uri(
            os.getenv("DATABASE_URI"),
            sample_rows_in_table_info=2  # Reduce sample rows to save tokens
        )
        logger.success("MariaDB database initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        exit(1)

    # Initialize the language model
    try:
        llm = initialize_llm(os.getenv("OPENAI_API_KEY"))
        logger.success("Language model initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize language model: {e}")
        exit(1)

    # Load SQL examples
    examples_file_path = os.getenv("EXAMPLES_FILE_PATH", "/Users/carrickcheah/llms_project/services/agent/vector/huggingtrain.jsonl")  # Use environment variable with fallback
    sql_examples = load_sql_examples(examples_file_path)

    # Create the SQLDatabaseToolkit with limited schema
    try:
        # Create the full toolkit first
        full_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        # Define tables relevant to your questions based on the examples
        # Extract table names from the SQL examples
        table_mentions = []
        for example in sql_examples:
            sql = example.get('sentence2_column', '').lower()
            for word in ['from', 'join']:
                if word in sql:
                    parts = sql.split(word)[1].strip().split()
                    if parts:
                        table = parts[0].strip(';,() ')
                        if table:
                            table_mentions.append(table)
        
        # Count occurrences and sort by most frequently mentioned
        from collections import Counter
        table_counts = Counter(table_mentions)
        relevant_tables = [table for table, _ in table_counts.most_common(15)]
        
        logger.info(f"Extracted tables from examples: {relevant_tables}")
        
        # Limit the schema to relevant tables
        limited_toolkit = limit_schema_size(
            full_toolkit, 
            max_tables=15,
            table_whitelist=relevant_tables
        )
        
        # Get tools from the limited toolkit
        tools = limited_toolkit.get_tools()
        logger.success("SQLDatabaseToolkit with limited schema initialized successfully!")

        # Initialize the SQL agent with the limited toolkit
        sql_agent = initialize_sql_agent(llm=llm, toolkit=limited_toolkit)
        
        # Create SQL generation prompt with examples
        schema_info = limited_toolkit.db.get_table_info()
        sql_generation_prompt = create_sql_generation_prompt(sql_examples, schema_info)
        sql_generation_chain = sql_generation_prompt | llm | StrOutputParser()
        logger.success("SQL generation chain initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to create SQLDatabaseToolkit or SQL agent: {e}")
        exit(1)

    # Pull the RAG prompt from LangChain Hub
    try:
        prompt = hub.pull("rlm/rag-prompt")
        logger.info("RAG prompt pulled successfully from LangChain Hub.")
    except Exception as e:
        logger.error(f"Failed to pull RAG prompt from LangChain Hub: {str(e)}")
        raise

    # Define document formatting function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Enhance prompt to handle domain-specific queries
    custom_prompt = prompt.partial(
        additional_context="If no specific correlation data is available, respond with: 'Based on the available data, I don't know if there is a correlation between rebate amount and shipping method chosen. However, recent studies and trends suggest that shipping fees and rebates can influence customer behavior, such as choosing faster or cheaper shipping methods based on cost incentives. You can explore web-based studies or trending discussions on X for more insights.'"
    )

    # Create LCEL retrieval-augmented generation chain
    try:
        qa_chain = (
            {
                "context": qdrant_store.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | custom_prompt
            | llm
            | StrOutputParser()
        )
        logger.success("LCEL RAG chain initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LCEL RAG chain: {str(e)}")
        raise

    # Improved answer_with_fallback function that prioritizes SQL execution for database questions
    def answer_with_fallback(question):
        try:
            # First, run the diagnostic test to check embeddings and Qdrant retrieval
            logger.info(f"Running diagnostic test for question: {question}")
            retrieved_docs = test_embedding_retrieval(question, qdrant_store, embeddings)
            
            # Determine if this is a database-related question
            if is_database_question(question):
                logger.info(f"Database question detected: {question}")
                
                # Skip RAG and go straight to SQL generation using retrieved examples
                if retrieved_docs:
                    try:
                        # Extract query examples from the retrieved documents
                        sql_examples = []
                        for doc in retrieved_docs:
                            try:
                                doc_data = json.loads(doc.page_content)
                                
                                # Try different formats
                                if "sentence1_column" in doc_data and "sentence2_column" in doc_data:
                                    example_q = doc_data.get("sentence1_column", "")
                                    example_sql = doc_data.get("sentence2_column", "")
                                elif "user_query" in doc_data and "generated_sql" in doc_data:
                                    example_q = doc_data.get("user_query", "")
                                    example_sql = doc_data.get("generated_sql", "")
                                else:
                                    continue
                                
                                sql_examples.append({
                                    "question": example_q,
                                    "sql": example_sql
                                })
                            except Exception as e:
                                logger.warning(f"Error parsing retrieved document: {str(e)}")
                                continue
                        
                        if sql_examples:
                            # Create a prompt specifically for this question using the retrieved examples
                            logger.info(f"Creating SQL generation prompt with {len(sql_examples)} retrieved examples")
                            
                            examples_text = "\n\n".join([
                                f"Similar Question: {ex['question']}\nCorresponding SQL: {ex['sql']}" 
                                for ex in sql_examples
                            ])
                            
                            retrieval_prompt = ChatPromptTemplate.from_template("""
                            You are an expert in converting natural language questions to SQL queries.
                            
                            I want to answer this question: {question}
                            
                            Here are similar questions with their corresponding SQL queries:
                            
                            {examples}
                            
                            Based on these examples, generate a SQL query that will answer my question.
                            Only return the SQL query without any explanation or comments.
                            """)
                            
                            # Generate SQL based on retrieved examples
                            logger.info("Generating SQL using retrieved examples...")
                            retrieval_chain = retrieval_prompt | llm | StrOutputParser()
                            sql_query = retrieval_chain.invoke({
                                "question": question,
                                "examples": examples_text
                            })
                            
                            # Clean up the response
                            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                            logger.info(f"Generated SQL using retrieved examples: {sql_query}")
                            
                            # Execute the generated SQL
                            try:
                                logger.info("Executing SQL query from retrieved examples...")
                                result = db.run(sql_query)
                                return f"SQL Database Answer (using retrieved examples):\nQuery: {sql_query}\n\nResult: {result}"
                            except Exception as sql_exec_error:
                                logger.warning(f"Error executing SQL from retrieved examples: {str(sql_exec_error)}")
                                # Continue to fallback methods
                        else:
                            logger.warning("No usable SQL examples found in retrieved documents")
                    except Exception as retrieval_error:
                        logger.warning(f"Error using retrieved examples: {str(retrieval_error)}")
                
                try:
                    # Fall back to generating SQL using the examples loaded from file
                    logger.info(f"Generating SQL query using file examples for: {question}")
                    sql_query = sql_generation_chain.invoke({"question": question})
                    logger.info(f"Generated SQL query from file examples: {sql_query}")
                    
                    # Execute the generated SQL
                    try:
                        result = db.run(sql_query)
                        return f"SQL Database Answer (using file examples):\nQuery: {sql_query}\n\nResult: {result}"
                    except Exception as sql_exec_error:
                        logger.warning(f"Error executing generated SQL: {str(sql_exec_error)}")
                        
                        # Fallback to SQL agent if execution fails
                        logger.info(f"Falling back to SQL agent for: {question}")
                        agent_response = sql_agent.invoke({"input": question})
                        
                        if isinstance(agent_response, dict) and "output" in agent_response:
                            return f"SQL Database Answer (via agent): {agent_response['output']}"
                        else:
                            return f"SQL Database Answer (via agent): {agent_response}"
                
                except Exception as sql_gen_error:
                    logger.error(f"SQL generation failed: {str(sql_gen_error)}")
                    return "I wasn't able to find relevant data to answer your question with the available tools."
            else:
                # For non-database questions, use the RAG chain
                logger.info(f"Non-database question detected. Trying to answer with RAG: {question}")
                response = qa_chain.invoke(question)
                
                if "I don't know" in response:
                    logger.info(f"RAG couldn't answer. Falling back to general response.")
                    return f"I don't have specific information about {question}. You might want to rephrase your question or ask about shipping methods, order data, or customer information that's available in the database."
                
                return f"Knowledge Base Answer: {response}"
                
        except Exception as e:
            logger.error(f"Error answering question '{question}': {str(e)}")
            return f"Error: Could not process the question due to: {str(e)}"

    # Example questions for demo (using questions from your examples)
    questions = [
        "Who is our top customer for Cycling Gloves in 2023?",
        "Who is our top customer for Yoga Mat in 2023?",
        "Who is our top customer for Skateboard all time?",
        "Who is our top customer for Agility Ladder in 2024?",
        "Generate all invoices for Bowling Ball in year 2024.",
        "What is the most used shipping method in France?",
        "What is the most used shipping method in China?",
        "What is the most used shipping method in Hong Kong?",
        "What is the most used shipping method in Malaysia?"
    ]

    # Run and print responses for questions
    for question in questions:
        print(f"\n> {question}")
        response = answer_with_fallback(question)
        print(response)