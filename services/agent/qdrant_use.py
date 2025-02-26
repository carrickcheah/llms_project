
from initialize_group import init_qdrant, dense_vector_search, hybrid_vector_search

# How to Use Qdrant

################################################################################
###                              Initial Setup                               ###
################################################################################

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# Initialize embedding model
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Load and process documents
loader = TextLoader("your_file.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1536, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Create the Qdrant collection
qdrant = init_qdrant(docs, embedding, collection_name="my_knowledge_base")


################################################################################
###               Later Searches (without reloading documents):              ###
################################################################################
# Search using dense vectors
results = dense_vector_search(
    query="What is machine learning?",
    embedding=embedding,
    collection_name="my_knowledge_base"
)

# Filter search by metadata
filtered_results = dense_vector_search(
    query="python examples",
    embedding=embedding,
    collection_name="my_knowledge_base",
    filter_condition={"must": [{"key": "category", "match": {"value": "programming"}}]}
)

# Try hybrid search for better results
hybrid_results = hybrid_vector_search(
    query="database optimization techniques",
    embedding=embedding,
    collection_name="my_knowledge_base"
)







