##########################################################################################
#    Initialization utilities for database, LLM, and vector store connections            #
##########################################################################################
#   Contents of table                                                                    #
#   1. Initialize Maria DB connection                                                    #
##########################################################################################

# Environment and database imports
from langchain_community.utilities import SQLDatabase
from loguru import logger

# Load environment variables once
from dotenv import load_dotenv
load_dotenv()
import os


#####################################################################################
####                           1.Init Maria DB connection                        ####
#####################################################################################
def initialize_database(database_uri: str) -> SQLDatabase:
    """Initialize and validate the database connection.
    
    Args:
        database_uri: URI for the database connection. Must be provided.
        
    Returns:
        SQLDatabase: Initialized database connection
        
    Raises:
        ValueError: If DATABASE_URI is not set
        ConnectionError: If connection to the database fails
    """
    if not database_uri:
        logger.error("DATABASE_URI is not set. Please provide a database URI.")
        raise ValueError("DATABASE_URI is not set.")
   
    try:
        logger.info(f"Connecting to database using URI: {database_uri}")
        db = SQLDatabase.from_uri(database_uri)
        # Test the connection by getting table info
        logger.info("Testing database connection by fetching usable table names...")
        db.get_usable_table_names()
        logger.success("Database connection validated successfully!")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to the database. Details: {e}")
        raise ConnectionError(f"Failed to connect to database: {str(e)}")