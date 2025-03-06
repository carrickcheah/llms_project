# Streamlit chat. It works

from typing import List
import os
import json
from loguru import logger
import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Preload embeddings to avoid PyTorch/Streamlit conflict
try:
    embeddings = HuggingFaceEmbeddings(model_name="carrick113/autotrain-wsucv-dqrgc")
    logger.info("HuggingFace embeddings preloaded successfully.")
except Exception as e:
    logger.error(f"Failed to preload embeddings: {str(e)}")
    raise

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "db" not in st.session_state:
    st.session_state.db = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "qdrant_store" not in st.session_state:
    st.session_state.qdrant_store = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = embeddings  # Use preloaded embeddings
if "sql_agent" not in st.session_state:
    st.session_state.sql_agent = None
if "sql_generation_chain" not in st.session_state:
    st.session_state.sql_generation_chain = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def initialize_llm(api_key: str) -> ChatOpenAI:
    """Initialize and validate the language model."""
    if not api_key:
        logger.error("api_key is not set. Please provide an API key.")
        raise ValueError("api_key is not set.")
   
    try:
        model = ChatOpenAI(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            request_timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2048")),
        )
        logger.success("Language model initialized successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize language model. Details: {e}")
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")

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

def create_sql_generation_prompt(examples, schema_info):
    """Create a prompt for SQL generation using examples."""
    few_shot_examples = ""
    for idx, example in enumerate(examples[:5]):
        question = example.get('sentence1_column', '')
        sql = example.get('sentence2_column', '')
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
    return any(keyword in question_lower for keyword in db_keywords)

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

def answer_with_fallback(question, qdrant_store, embeddings, db, sql_agent, sql_generation_chain, qa_chain):
    """Process a question and return a response with SQL query and result if applicable."""
    try:
        logger.info(f"Running diagnostic test for question: {question}")
        retrieved_docs = test_embedding_retrieval(question, qdrant_store, embeddings)
        
        if is_database_question(question):
            logger.info(f"Database question detected: {question}")
            
            if retrieved_docs:
                try:
                    sql_examples = []
                    for doc in retrieved_docs:
                        try:
                            example_q = doc.page_content
                            example_sql = doc.metadata.get("sql_query", "") if isinstance(doc.metadata, dict) else str(doc.metadata)
                            
                            if example_q and example_sql:
                                sql_examples.append({
                                    "question": example_q,
                                    "sql": example_sql
                                })
                        except Exception as e:
                            logger.warning(f"Error parsing retrieved document: {str(e)}")
                            continue
                    
                    if sql_examples:
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
                        
                        logger.info("Generating SQL using retrieved examples...")
                        # Fix: Use st.session_state.llm instead of sql_agent.llm
                        retrieval_chain = retrieval_prompt | st.session_state.llm | StrOutputParser()
                        sql_query = retrieval_chain.invoke({
                            "question": question,
                            "examples": examples_text
                        })
                        
                        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                        logger.info(f"Generated SQL using retrieved examples: {sql_query}")
                        
                        try:
                            logger.info("Executing SQL query from retrieved examples...")
                            result = db.run(sql_query)
                            return {
                                "answer": f"SQL Database Answer (using retrieved examples): {result}",
                                "query": sql_query,
                                "result": str(result)
                            }
                        except Exception as sql_exec_error:
                            logger.warning(f"Error executing SQL from retrieved examples: {str(sql_exec_error)}")
                            return {"error": f"Error executing SQL: {str(sql_exec_error)}"}
                    else:
                        logger.warning("No usable SQL examples found in retrieved documents")
                except Exception as retrieval_error:
                    logger.warning(f"Error using retrieved examples: {str(retrieval_error)}")
                    return {"error": f"Error processing retrieved examples: {str(retrieval_error)}"}
            
            try:
                logger.info(f"Generating SQL query using file examples for: {question}")
                sql_query = sql_generation_chain.invoke({"question": question})
                logger.info(f"Generated SQL query from file examples: {sql_query}")
                
                try:
                    result = db.run(sql_query)
                    return {
                        "answer": f"SQL Database Answer (using file examples): {result}",
                        "query": sql_query,
                        "result": str(result)
                    }
                except Exception as sql_exec_error:
                    logger.warning(f"Error executing generated SQL: {str(sql_exec_error)}")
                    
                    logger.info(f"Falling back to SQL agent for: {question}")
                    agent_response = sql_agent.invoke({"input": question})
                    
                    if isinstance(agent_response, dict) and "output" in agent_response:
                        return {
                            "answer": f"SQL Database Answer (via agent): {agent_response['output']}",
                            "query": "Agent-generated query not explicitly available",
                            "result": str(agent_response['output'])
                        }
                    else:
                        return {
                            "answer": f"SQL Database Answer (via agent): {agent_response}",
                            "query": "Agent-generated query not explicitly available",
                            "result": str(agent_response)
                        }
            
            except Exception as sql_gen_error:
                logger.error(f"SQL generation failed: {str(sql_gen_error)}")
                return {"error": "I wasn't able to find relevant data to answer your question with the available tools."}
        else:
            logger.info(f"Non-database question detected. Trying to answer with RAG: {question}")
            response = qa_chain.invoke(question)
            
            if "I don't know" in response:
                logger.info(f"RAG couldn't answer. Falling back to general response.")
                return {
                    "answer": f"I don't have specific information about {question}. You might want to rephrase your question or ask about shipping methods, order data, or customer information that's available in the database.",
                    "query": None,
                    "result": None
                }
            
            return {
                "answer": f"Knowledge Base Answer: {response}",
                "query": None,
                "result": None
            }
            
    except Exception as e:
        logger.error(f"Error answering question '{question}': {str(e)}")
        return {"error": f"Error: Could not process the question due to: {str(e)}"}

def initialize_agent():
    """Initialize all components and store them in session state."""
    with st.spinner("Initializing agent..."):
        warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

        # Use preloaded embeddings
        st.session_state.embeddings = embeddings

        # Connect to Qdrant
        try:
            st.session_state.qdrant_store = use_existing_qdrant(
                embedding=st.session_state.embeddings,
                url="http://localhost:6333",
                collection_name="query_embeddings",
                prefer_grpc=True
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant vector store: {str(e)}")
            st.error(f"Failed to connect to Qdrant: {str(e)}")
            return

        # Initialize database
        try:
            st.session_state.db = SQLDatabase.from_uri(
                os.getenv("DATABASE_URI"),
                sample_rows_in_table_info=2
            )
            logger.success("MariaDB database initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            st.error(f"Failed to initialize database: {e}")
            return

        # Initialize LLM
        try:
            st.session_state.llm = initialize_llm(os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            st.error(f"Failed to initialize LLM: {e}")
            return

        # Load SQL examples
        examples_file_path = os.getenv("EXAMPLES_FILE_PATH", "/Users/carrickcheah/llms_project/services/agent/vector/efg.jsonl")
        sql_examples = load_sql_examples(examples_file_path)

        # Initialize SQL toolkit and agent
        try:
            full_toolkit = SQLDatabaseToolkit(db=st.session_state.db, llm=st.session_state.llm)
            
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
            
            from collections import Counter
            table_counts = Counter(table_mentions)
            relevant_tables = [table for table, _ in table_counts.most_common(15)]
            
            logger.info(f"Extracted tables from examples: {relevant_tables}")
            
            limited_toolkit = limit_schema_size(
                full_toolkit, 
                max_tables=15,
                table_whitelist=relevant_tables
            )
            
            logger.success("SQLDatabaseToolkit with limited schema initialized successfully!")
            st.session_state.sql_agent = initialize_sql_agent(llm=st.session_state.llm, toolkit=limited_toolkit)
            
            schema_info = limited_toolkit.db.get_table_info()
            sql_generation_prompt = create_sql_generation_prompt(sql_examples, schema_info)
            st.session_state.sql_generation_chain = sql_generation_prompt | st.session_state.llm | StrOutputParser()
            logger.success("SQL generation chain initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to create SQLDatabaseToolkit or SQL agent: {e}")
            st.error(f"Failed to initialize SQL agent: {e}")
            return

        # Initialize RAG
        try:
            prompt = hub.pull("rlm/rag-prompt")
            logger.info("RAG prompt pulled successfully from LangChain Hub.")
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            custom_prompt = prompt.partial(
                additional_context="If no specific correlation data is available, respond with: 'Based on the available data, I don't know if there is a correlation between rebate amount and shipping method chosen. However, recent studies and trends suggest that shipping fees and rebates can influence customer behavior, such as choosing faster or cheaper shipping methods based on cost incentives. You can explore web-based studies or trending discussions on X for more insights.'"
            )

            st.session_state.qa_chain = (
                {
                    "context": st.session_state.qdrant_store.as_retriever() | format_docs,
                    "question": RunnablePassthrough(),
                }
                | custom_prompt
                | st.session_state.llm
                | StrOutputParser()
            )
            logger.success("LCEL RAG chain initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LCEL RAG chain: {str(e)}")
            st.error(f"Failed to initialize RAG chain: {str(e)}")
            return

        st.session_state.initialized = True
        st.success("Agent initialized successfully!")

# Sidebar for initialization
with st.sidebar:
    st.header("Agent Configuration")
    # Add option to disable file watching to reduce PyTorch warnings
    disable_watcher = st.checkbox("Disable File Watcher (Reduce Warnings)", value=False)
    if st.button("Initialize Agent"):
        if disable_watcher:
            os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # Disable watcher
            logger.info("File watcher disabled to reduce PyTorch/Streamlit conflicts.")
        initialize_agent()

# Main chat interface
st.title("🤖 SQL Agent Chat")
st.caption("Query your database using natural language")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "query" in message and message["query"]:
            with st.expander("View SQL Query"):
                st.code(message["query"], language="sql")
        if "result" in message and message["result"]:
            with st.expander("View Raw SQL Results"):
                st.text(message["result"])

# Input for new questions
if prompt := st.chat_input("Ask a question about your database..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check if initialized
    if not st.session_state.initialized:
        with st.chat_message("assistant"):
            st.write("Please initialize the agent first using the sidebar.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Please initialize the agent first using the sidebar."
            })
    else:
        # Process the question
        with st.spinner("Thinking..."):
            state = answer_with_fallback(
                prompt, 
                st.session_state.qdrant_store, 
                st.session_state.embeddings, 
                st.session_state.db, 
                st.session_state.sql_agent, 
                st.session_state.sql_generation_chain, 
                st.session_state.qa_chain
            )
        
        # Display assistant response
        with st.chat_message("assistant"):
            if state.get("error"):
                st.error(state["error"])
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Error: {state['error']}"
                })
            else:
                st.write(state["answer"])
                if state["query"]:
                    with st.expander("View SQL Query"):
                        st.code(state["query"], language="sql")
                if state["result"]:
                    with st.expander("View Raw SQL Results"):
                        st.text(state["result"])
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": state["answer"],
                    "query": state["query"],
                    "result": state["result"]
                })

# Reset file watcher to default after initialization (optional)
if st.session_state.initialized and "STREAMLIT_SERVER_FILE_WATCHER_TYPE" in os.environ:
    del os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"]