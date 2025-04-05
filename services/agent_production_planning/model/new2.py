from openai import AsyncOpenAI
from agents import set_default_openai_client, Agent, ModelSettings, function_tool

custom_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
set_default_openai_client(custom_client)

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    print(f"\nTool called: get_weather({city})")
    return f"The weather in {city} is sunny"

# Create the agent
agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="gemma3",
    tools=[get_weather],
)

# Print agent configuration details
print("\n===== Agent Configuration =====")
print(f"Name: {agent.name}")
print(f"Model: {agent.model}")
print(f"Instructions: {agent.instructions}")
print(f"Tools count: {len(agent.tools)}")
print(f"First tool: {agent.tools[0]}")

# Print agent attributes for debugging
print("\n===== Agent Details =====")
for attr in dir(agent):
    if not attr.startswith('__'):
        try:
            value = getattr(agent, attr)
            if not callable(value):
                print(f"{attr}: {value}")
        except Exception as e:
            print(f"{attr}: Error getting value: {e}")
            
print("\n===== Agent Methods =====")
for attr in dir(agent):
    if not attr.startswith('__') and callable(getattr(agent, attr)):
        print(f"{attr}()")

print("\nAgent initialization complete!")
