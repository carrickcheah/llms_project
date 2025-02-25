"""
This script is for toolbox.
TI want keep the while true 
"""


import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from toolbox_langchain import ToolboxClient
from langgraph.errors import GraphRecursionError  # Import specific error if needed

prompt = """
You're a helpful hotel assistant. You handle hotel searching, booking and
cancellations. When the user searches for a hotel, mention its name, id, 
location and price tier. Always mention hotel ids while performing any 
searches. This is very important for any operations. For any bookings or 
cancellations, please provide the appropriate confirmation. Be sure to 
update checkin or checkout dates if mentioned by the user.
Don't ask for confirmations from the user.
"""

def main():
    # Initialize the model
    model = ChatOllama(model="llama3.1", temperature=0)
    
    # Load tools from the Toolbox server
    client = ToolboxClient("http://127.0.0.1:5000")
    tools = client.load_toolset()

    # Create the agent with a memory checkpoint
    agent = create_react_agent(model, tools, checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "thread-1"}}

    print("Welcome Boss! What can I help with?")
    while True:
        user_query = input(">> ").strip()
        if not user_query:
            continue  # Skip if no input is provided
        
        # Combine the system prompt with the user input
        inputs = {"messages": [("user", prompt + "\n" + user_query)]}

        try:
            # Invoke the agent and retrieve the response
            response = agent.invoke(inputs, stream_mode="values", config=config)
            output_message = response["messages"][-1].content
        except (GraphRecursionError, Exception):
            # Fallback message in case of errors or if the answer is not found.
            output_message = "Information not available. Please contact the database manager."
        
        print("Response:", output_message)

if __name__ == "__main__":
    main()
