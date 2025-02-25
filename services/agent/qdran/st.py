from sentence_transformers import SentenceTransformer
import json
import numpy as np

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load your abc.json file
with open('abc.json', 'r') as f:
    tools = json.load(f)

# Function to generate embeddings for a tool's description and parameters
def generate_embedding(tool):
    # Combine the description and parameter descriptions into a single string
    text_to_embed = tool['description'] + " " + " ".join([param['description'] for param in tool['parameters']])
    
    # Generate the embedding for the combined text
    embedding = model.encode(text_to_embed)
    
    return embedding.tolist()  # Convert numpy array to list for JSON serialization

for tool in tools["tools"]:
    tool['embedding'] = generate_embedding(tool)


# Save the updated JSON with embeddings back to a file
with open('abc_with_embeddings.json', 'w') as f:
    json.dump(tools, f, indent=4)

print("Embeddings pre-computed and saved to 'abc_with_embeddings.json'")












