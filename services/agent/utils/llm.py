##########################################################################################
#    Initialization utilities for database, LLM, and vector store connections            #
##########################################################################################
#   Contents of table                                                                    #                                               #
#   1. Initialize OPENAI GPT-4o-mini                                                     #       
#   2. Initialize Deepseek V3/ R1 model                                                  #
#   3. Init Lanchain format Deepseek                                                    #
##########################################################################################
import os
from loguru import logger
from typing import Optional
# Environment and database imports
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

# Load environment variables once
from dotenv import load_dotenv
load_dotenv()


#####################################################################################
####                        1. Init OPENAI GPT-4o-mini                           ####
#####################################################################################
def initialize_llm(api_key: str) -> ChatOpenAI:
    """Initialize and validate the language model.
    
    Args:
        api_key: OpenAI API key. Must be provided.
        
    Returns:
        ChatOpenAI: Initialized language model
        
    Raises:
        ValueError: If api_key is not set
        ConnectionError: If initialization of the language model fails
    """
    if not api_key:
        logger.error("api_key is not set. Please provide an API key.")
        raise ValueError("api_key is not set.")
   
    try:
        model = ChatOpenAI(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # Default to "gpt-4" if not set
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),  # Default to 0.0
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),  # Default to 3
            request_timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),  # Default to 30 seconds
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2048")),  # Default to 2048 tokens
        )
        logger.success("Language model initialized successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize language model. Details: {e}")
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")
    
#####################################################################################
####                         2. Init Deepseek V3/ R1 model                       ####
#####################################################################################

def initialize_deepseek(api_key: Optional[str] = None) -> ChatDeepSeek:
    """
    Initialize and validate the DeepSeek language model using environment variables.

    Args:
        api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.

    Returns:
        ChatDeepSeek: Initialized language model

    Raises:
        ValueError: If DEEPSEEK_API_KEY is not set
        ConnectionError: If initialization of the language model fails
    """
    # Use provided API key or get from environment variable
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_URL")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set.")
    
    if not base_url:
        raise ValueError("DEEPSEEK_URL is not set.")

    try:
        model = ChatDeepSeek(
            api_key=api_key,
            model=model_name,
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
            request_timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "30")),
            max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048")),
            base_url=base_url
        )
        return model
    except Exception as e:
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")
    

# #####################################################################################
# ####                  3     Init Lanchain format Deepseek                       ####
# #####################################################################################

# # def initialize_llm(api_key: str = None) -> ChatOpenAI:
# #     """Initialize and validate the language model using OpenAI client with DeepSeek API.
    
# #     Args:
# #         api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.
        
# #     Returns:
# #         ChatOpenAI: Initialized language model configured for DeepSeek
        
# #     Raises:
# #         ValueError: If DEEPSEEK_API_KEY is not set
# #         ConnectionError: If initialization of the language model fails
# #     """
# #     # Use provided API key or get from environment variable
# #     api_key = os.getenv("DEEPSEEK_API_KEY")
# #     base_url = os.getenv("DEEPSEEK_URL")
# #     model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
   
# #     if not api_key:
# #         raise ValueError("DEEPSEEK_API_KEY is not set.")
    
# #     if not base_url:
# #         raise ValueError("DEEPSEEK_URL is not set.")
   
# #     try:
# #         model = ChatOpenAI(
# #             api_key=api_key,
# #             model=model_name,
# #             temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0")),
# #             max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
# #             request_timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "30")),
# #             max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048")),
# #             base_url=base_url
# #         )
# #         return model
# #     except Exception as e:
# #         raise ConnectionError(f"Failed to initialize language model: {str(e)}")
# #####################################################################################
# #####################################################################################










