##########################################################################################
#    Initialization utilities for database, LLM, and vector store connections            #
##########################################################################################
#   Contents of table                                                                    #                                              #
#   4. Qdrant Initialization and other Functions                                         #
#   5. Get existing Qdrant store                                                         #
#   6. Init Qdrant from text                                                             #
#   7. Dense vector search                                                               #
#   8. Sparse vector search                                                              #
#   9. Hybrid vector search                                                              #
#   10. Convenience function to create a search function based on the mode               #                                                 #
##########################################################################################


from langchain_qdrant import QdrantVectorStore
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_qdrant import RetrievalMode, FastEmbedSparse

from loguru import logger

# Load environment variables once
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="carrick113/autotrain-u1whw-a7kfd")


doc_store = QdrantVectorStore.from_texts(
    texts, embeddings, url="<qdrant-url>", api_key="<qdrant-api-key>", collection_name="texts"
)










