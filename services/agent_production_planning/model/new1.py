from ollama import Client

client = Client(host='http://localhost:11434')

response = client.show('xingyaow/codeact-agent-mistral')

print(response)