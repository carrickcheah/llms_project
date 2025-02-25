import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai

# Load JSON data
with open("text2sql_dataset.json", "r") as f:
    text2sql_dataset = json.load(f)

# 1. Connect to Qdrant
qdrant = QdrantClient("http://localhost:6333")

# 2. Create or recreate a new collection
collection_name = "text2sql_queries"

# Check if the collection exists, and delete it if it does
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name)

# Create a new collection
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance="Cosine")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_text2sql_pairs(dataset, collection_name):
    points = []
    for idx, item in enumerate(dataset):
        user_query = item["user_input"]
        sql_query = item["generated_sql"]

        # 1. Embed the user_input
        embedding = model.encode(user_query).tolist()

        # 2. Create a point for Qdrant
        payload = {
            "user_query": user_query,
            "generated_sql": sql_query
        }
        point = PointStruct(id=idx, vector=embedding, payload=payload)

        points.append(point)

    # 3. Upsert into Qdrant
    qdrant.upsert(collection_name=collection_name, points=points)
    print("✅ Ingested Text2SQL data into Qdrant!")

# Call the ingestion function
ingest_text2sql_pairs(text2sql_dataset, collection_name)

def retrieve_similar_sql(user_query, collection_name, top_k=3):
    # 1. Embed the new user query
    query_vector = model.encode(user_query).tolist()

    # 2. Search Qdrant using `search`
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    # 3. Format the result
    similar_matches = []
    for r in results:
        similar_matches.append({
            "score": r.score,
            "matched_query": r.payload["user_query"],
            "matched_sql": r.payload["generated_sql"]
        })
    return similar_matches

# Example retrieval
new_user_query = "Which customers rented movies last month?"
similar_queries = retrieve_similar_sql(new_user_query, collection_name)

for match in similar_queries:
    print(f"Score: {match['score']:.4f}")
    print(f"User Query: {match['matched_query']}")
    print(f"SQL: {match['matched_sql']}")
    print("---")
