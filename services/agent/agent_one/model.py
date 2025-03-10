# model.py #  never edit this row
##########################################################################################
#    Initialization utilities for database, LLM, and vector store connections            #
##########################################################################################
#   Contents of table                                                                    #                                               #
#   1. Initialize OPENAI GPT-4o-mini                                                     #       
#   2. Initialize Deepseek V3/ R1 model                                                  #
#   3. Init Lanchain format Deepseek                                                     #
#   4. Initialize Anthropic Claude 3.5 Haiku                                             #
##########################################################################################
import os
from loguru import logger
from typing import Optional

# Environment and database imports
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables once
from dotenv import load_dotenv
load_dotenv()

##########################################################################################
#                        1. Initialize OpenAI GPT-4o                                     #
##########################################################################################

def initialize_openai(api_key: str) -> ChatOpenAI:
    """Initialize and validate the language model."""
    if not api_key:
        logger.error("api_key is not set. Please provide an API key.")
        raise ValueError("api_key is not set.")
   
    try:
        model = ChatOpenAI(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            request_timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2048")),
        )
        logger.success(f"Language model '{model.model_name}' initialized successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize language model. Details: {e}")
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")

##########################################################################################
#                        2. Initialize DeepSeek Chat Model                               #
##########################################################################################

def initialize_deepseek(api_key: str) -> ChatDeepSeek:
    """Initialize DeepSeek model with environment-configured parameters.

    Args:
        api_key: DeepSeek API key (required).

    Returns:
        ChatDeepSeek: Initialized DeepSeek model.

    Raises:
        ValueError: If api_key is not provided.
        ConnectionError: If initialization fails (e.g., network or API issues).
    """
    # Use environment variable if set, otherwise default to https://api.deepseek.com/v1
    base_url = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        logger.error("Missing DeepSeek API Key.")
        raise ValueError("DeepSeek API key is required.")

    try:
        model = ChatDeepSeek(
            api_key=api_key,
            model=model_name,
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
            timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "30")),
            max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048")),
            base_url=base_url
        )
        logger.success(f"Language model '{model.model_name}' initialized successfully!")
        return model
    except Exception as e:
        logger.error(f"DeepSeek model initialization failed: {e}")
        raise ConnectionError(f"Failed to initialize DeepSeek model: {str(e)}")



# #####################################################################################
# ####                  3     Init Lanchain format Deepseek                       ####
# #####################################################################################


def initialize_open_deep(api_key: Optional[str] = None) -> ChatOpenAI:
    """Initialize and validate the language model using OpenAI client with DeepSeek API.
    
    Args:
        api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.
        
    Returns:
        ChatOpenAI: Initialized language model configured for DeepSeek
        
    Raises:
        ValueError: If DEEPSEEK_API_KEY is not set
        ConnectionError: If initialization of the language model fails
    """
    # Use provided API key or get from environment variable
    base_url = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
   
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set.")
    
    if not base_url:
        raise ValueError("DEEPSEEK_URL is not set.")
   
    try:
        model = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
            request_timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "30")),
            max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048")),
            base_url=base_url
        )
        logger.success(f"Language model '{model.model_name}' initialized successfully!")
        return model
    except Exception as e:
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")

##########################################################################################
#                     4. Initialize Anthropic Claude 3.5 Haiku                           #
##########################################################################################


def initialize_anthropic(api_key: Optional[str] = None) -> ChatAnthropic:
    """Initialize Claude 3.5 Haiku with environment-configured parameters."""

    if not api_key:
        logger.error("Missing Anthropic API Key.")
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    try:
        model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        model = ChatAnthropic(
            api_key=api_key,
            model=model_name,
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("ANTHROPIC_MAX_RETRIES", "3")),
            request_timeout=int(os.getenv("ANTHROPIC_TIMEOUT", "30")),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048")),
        )
        logger.success(f"Language model '{model_name}' initialized successfully!")
        return model
    except Exception as e:
        logger.error(f"Claude model initialization failed: {e}")
        raise ConnectionError(f"Failed to initialize Claude model: {str(e)}")


##########################################################################################
#                          5.  Embedding model:  Huggingface                             #
##########################################################################################


def initialize_embeddings(model_name: str = "carrick113/autotrain-wsucv-dqrgc") -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embeddings with the specified model.

    Args:
        model_name: Name of the HuggingFace model to load (default: "carrick113/autotrain-wsucv-dqrgc").

    Returns:
        HuggingFaceEmbeddings: Initialized embeddings object.

    Raises:
        Exception: If embedding initialization fails (e.g., model not found, network error).
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.success(f"HuggingFace embeddings preloaded successfully with model: {model_name}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to preload embeddings for model '{model_name}': {str(e)}")
        raise


# #####################################################################################
# ####                  5     Nvidia format Deepseek                       ####
# #####################################################################################

def initialize_nvidia_deep(api_key: Optional[str] = None) -> ChatOpenAI:
    """
    Initialize and validate the language model using OpenAI client with DeepSeek API.
    
    Args:
        api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.
        
    Returns:
        ChatOpenAI: Initialized language model configured for DeepSeek
        
    Raises:
        ValueError: If DEEPSEEK_API_KEY is not set
        ConnectionError: If initialization of the language model fails
    """
    # Use provided API key or get from environment variable
    base_url = "https://integrate.api.nvidia.com/v1"  # Remove the trailing comma to make it a string
    model_name = "deepseek-ai/deepseek-r1"  # Remove the trailing comma to make it a string

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set.")
    
    if not base_url:
        raise ValueError("DEEPSEEK_URL is not set.")
   
    try:
        model = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0,
            max_retries=3,
            request_timeout=30,
            max_tokens=4096,
            base_url=base_url,
            # stream=True
        )
        logger.success(f"NVIDIA model '{model.model_name}' initialized successfully!")
        return model
    except Exception as e:
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")








