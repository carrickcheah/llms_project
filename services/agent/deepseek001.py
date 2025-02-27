from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the OpenAI client with API key and base URL for DeepSeek
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

# Create client with DeepSeek's base URL
client = OpenAI(
    api_key=deepseek_api_key,  # Use DEEPSEEK_API_KEY, not OPENAI_API_KEY
    base_url="https://api.deepseek.com/v1"  # Make sure to use the correct API version path
)

# Define the chat completion request
def get_chat_response(messages, model="deepseek-chat", stream=False):
    """
    Sends a chat completion request to the DeepSeek API.

    Args:
        messages (list): List of message dictionaries (e.g., [{"role": "user", "content": "Hello"}]).
        model (str): The model to use for the chat completion.
        stream (bool): Whether to stream the response.

    Returns:
        str: The content of the assistant's response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {str(e)}")
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant that translates English to French."},
        {"role": "user", "content": "I love programming."},
    ]
    
    # Print request information for debugging
    print(f"Sending request to DeepSeek API with model: {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}")
    
    # Get response
    response = get_chat_response(messages, model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    print("\nResponse from DeepSeek:")
    print(response)