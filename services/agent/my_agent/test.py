from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain.schema import HumanMessage  # Use HumanMessage for agent input

# Define tools if available; otherwise, leave as empty list.
tools = []

# Initialize database connection
DATABASE_URI = "mysql+pymysql://myuser:mypassword@127.0.0.1:3306/nex_valiant"
db = SQLDatabase.from_uri(DATABASE_URI)

# Pull down the text-to-SQL prompt (likely a ChatPromptTemplate)
text_to_sql_prompt = hub.pull("rlm/text-to-sql")
# Convert the prompt to a string – use its template attribute if available.
if hasattr(text_to_sql_prompt, "template"):
    system_prompt_str = text_to_sql_prompt.template
else:
    system_prompt_str = str(text_to_sql_prompt)

# Initialize the ChatOpenAI model
model = ChatOpenAI()

# Create a chain using LangChain Expression Language for SQL query generation
chain_inputs = {
    "table_info": lambda x: db.get_table_info(),
    "input": lambda x: x["question"],
    "few_shot_examples": lambda x: "",
    "dialect": lambda x: db.dialect,
}
sql_response_chain = (
    chain_inputs
    | text_to_sql_prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

def main():
    """
    Main function to run the interactive agent.
    """
    # Create the agent with a memory checkpoint.
    agent = create_react_agent(model, tools, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "thread-1"}}
    
    print("Welcome Boss! What can I help with?")
    while True:
        user_query = input(">> ").strip()
        if not user_query:
            continue  # Skip if no input is provided
        
        # Create a proper HumanMessage using the system prompt (as context) and the user query.
        message_text = system_prompt_str + "\n" + user_query
        agent_input = {"messages": [HumanMessage(content=message_text)]}
        
        try:
            # Invoke the agent and retrieve the response in stream mode.
            response = agent.invoke(agent_input, stream_mode="values", config=config)
            output_message = response["messages"][-1].content
        except (GraphRecursionError, Exception) as e:
            output_message = "Information not available. Please contact the database manager."
            print(f"Error encountered: {e}")
        
        print("Response:", output_message)

if __name__ == "__main__":
    main()
