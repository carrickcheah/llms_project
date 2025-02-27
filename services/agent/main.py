from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

def initialize_llm(api_key: str = None) -> ChatOpenAI:
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
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_URL")
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
        return model
    except Exception as e:
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        print(f"Initializing DeepSeek with model: {os.getenv('DEEPSEEK_MODEL')}")
        print(f"Using base URL: {os.getenv('DEEPSEEK_URL')}")
        
        llm = initialize_llm()
        
        messages = [
            ("system", "You are a helpful assistant that translates English to Chinese. Translate the user sentence."),
            ("human", "I love pretty and sexy women."),
        ]
        
        ai_msg = llm.invoke(messages)
        print("\nTranslation result:")
        print(ai_msg.content)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()