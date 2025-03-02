from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import json
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType
import os
import warnings
from loguru import logger
from llm import initialize_llm

# Load environment variables
load_dotenv()

# Configure loguru (no need for basicConfig)
logger.add("app.log", rotation="10 MB", level="INFO")

class QdrantSettings(BaseSettings):
    """Configuration settings for Qdrant."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="QDRANT_"
    )
    url: str = "http://localhost:6333"
    collection_name: str = "query_embeddings"
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    content_payload_key: str = "full_json"
    hf_model_name: str = "carrick113/autotrain-wsucv-dqrgc"
    file_path: str = "/path/to/your/abc.json"

def get_embeddings(settings: QdrantSettings) -> Embeddings:
    """Initialize Hugging Face embeddings."""
    return HuggingFaceEmbeddings(model_name=settings.hf_model_name)

def get_qdrant_client(settings: QdrantSettings) -> QdrantClient:
    """Create and return Qdrant client."""
    return QdrantClient(
        url=settings.url,
        prefer_grpc=settings.prefer_grpc,
        api_key=settings.api_key
    )

def load_documents(file_path: str) -> List[Document]:
    """Load and validate documents from JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of query objects")

    documents = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a dictionary")

        # Structure document content
        content = {
            "question": item["user_query"],
            "answer": {
                "sql": item["generated_sql"],
                "tables": item["tables_used"],
                "columns": item["columns_used"]
            }
        }
        metadata = {
            "schema": item.get("schema_name", "unknown"),
            "executable": item.get("execute_in_mysql", False)
        }
        documents.append(Document(
            page_content=json.dumps(content),
            metadata=metadata
        ))
    return documents

def connect_vector_store(
    settings: QdrantSettings,
    embeddings: Embeddings
) -> QdrantVectorStore:
    """Connect to or initialize Qdrant vector store."""
    client = get_qdrant_client(settings)
    
    try:
        # Check for existing collection
        collections = client.get_collections().collections
        if any(c.name == settings.collection_name for c in collections):
            logger.info(f"Using existing collection: {settings.collection_name}")
            return QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=settings.collection_name,
                url=settings.url,
                api_key=settings.api_key,
                content_payload_key=settings.content_payload_key
            )
    except Exception as e:
        logger.warning(f"Collection check failed: {e}")

    # Create new collection with documents
    logger.info(f"Initializing new collection: {settings.collection_name}")
    documents = load_documents(settings.file_path)
    
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=settings.url,
        collection_name=settings.collection_name,
        api_key=settings.api_key,
        content_payload_key=settings.content_payload_key
    )

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

def answer_with_fallback(question, qdrant_store, db, llm, sql_agent):
    """Answer questions with fallback mechanisms."""
    try:
        # First, run the diagnostic test to check embeddings and Qdrant retrieval
        logger.info(f"Running diagnostic test for question: {question}")
        retrieved_docs = qdrant_store.as_retriever().invoke(question)
        
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

def main():
    """Main application entry point."""
    try:
        # Initialize settings and embeddings
        settings = QdrantSettings()
        embeddings = get_embeddings(settings)
        
        # Connect to Qdrant vector store
        qdrant_store = connect_vector_store(settings, embeddings)
        
        # Initialize the database
        db = SQLDatabase.from_uri(
            os.getenv("DATABASE_URI"),
            sample_rows_in_table_info=2  # Reduce sample rows to save tokens
        )
        logger.success("MariaDB database initialized successfully!")
        
        # Initialize the language model
        llm = initialize_llm(os.getenv("OPENAI_API_KEY"))
        logger.success("Language model initialized successfully!")
        
        # Create the SQLDatabaseToolkit with limited schema
        full_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        limited_toolkit = limit_schema_size(full_toolkit, max_tables=15)
        
        # Initialize the SQL agent
        sql_agent = initialize_sql_agent(llm=llm, toolkit=limited_toolkit)
        
        # Pull the RAG prompt from LangChain Hub
        prompt = hub.pull("rlm/rag-prompt")
        logger.info("RAG prompt pulled successfully from LangChain Hub.")
        
        # Create LCEL retrieval-augmented generation chain
        qa_chain = (
            {
                "context": qdrant_store.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.success("LCEL RAG chain initialized successfully.")
        
        # Example questions for demo
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
            response = answer_with_fallback(question, qdrant_store, db, llm, sql_agent)
            print(response)
            
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()