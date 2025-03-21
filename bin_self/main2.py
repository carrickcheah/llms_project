# this is file
# import os
# from typing import Optional
# from langchain_openai import ChatOpenAI
# from loguru import logger

# def initialize_open_deep(api_key: Optional[str] = None) -> ChatOpenAI:
#     """Initialize and validate the language model using OpenAI client with DeepSeek API.
    
#     Args:
#         api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.
        
#     Returns:
#         ChatOpenAI: Initialized language model configured for DeepSeek
        
#     Raises:
#         ValueError: If DEEPSEEK_API_KEY is not set
#         ConnectionError: If initialization of the language model fails
#     """


#     good
#     # Retrieve API key and base URL from environment variables
#     api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
#     base_url = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1")
#     model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

#     if not api_key:
#         raise ValueError("DEEPSEEK_API_KEY is not set.")
    
#     if not base_url:
#         raise ValueError("DEEPSEEK_URL is not set.")

#     try:
#         model = ChatOpenAI(
#             api_key=api_key,
#             model=model_name,
#             temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0")),
#             max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
#             request_timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "30")),
#             max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048")),
#             base_url=base_url
#         )
#         logger.success(f"Language model '{model_name}' initialized successfully!")
#         return model
#     except Exception as e:
#         logger.error(f"DeepSeek model initialization failed: {e}")
#         raise ConnectionError(f"Failed to initialize DeepSeek model: {str(e)}")

# def chat_with_model(model: ChatOpenAI):
#     """Interactive chat function for user input."""
#     print("\n💬 Interactive DeepSeek Chat. Type 'exit' to stop.\n")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Exiting chat... 👋")
#             break
#         try:
#             response = model.invoke(user_input)
#             print(f"DeepSeek: {response.content}\n")
#         except Exception as e:
#             print(f"Error: {str(e)}")

# if __name__ == "__main__":
#     model = initialize_open_deep()
#     chat_with_model(model)
