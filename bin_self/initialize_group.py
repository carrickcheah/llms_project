# ##########################################################################################
# #    Initialization utilities for database, LLM, and vector store connections            #
# ##########################################################################################                                                 #
# #   4. Qdrant Initialization and other Functions                                         #
# #   5. Get existing Qdrant store                                                         #
# #   6. Init Qdrant from text                                                             #
# #   7. Dense vector search                                                               #
# #   8. Sparse vector search                                                              #
# #   9. Hybrid vector search                                                              #
# #   10. Convenience function to create a search function based on the mode               #
#                                          #
# ##########################################################################################
# # Qdrant imports
# from langchain_qdrant import QdrantVectorStore
# from langchain.embeddings.base import Embeddings
# from langchain.schema import Document
# from langchain_qdrant import RetrievalMode, FastEmbedSparse
# from loguru import logger

# # Load environment variables once
# from dotenv import load_dotenv
# load_dotenv()




# #####################################################################################
# ####              4.   Qdrant Initialization and other Functions                 ####
# #####################################################################################

# # Initialize Qdrant vector store with documents
# def init_qdrant(
#     docs: List[Document],
#     embedding: Embeddings,
#     url: str = "http://localhost:6333",
#     collection_name: str = "query_embeddings",
#     prefer_grpc: bool = True,
#     overwrite: bool = True
# ) -> QdrantVectorStore:
#     """Initialize a Qdrant vector store with the given documents.
#     If the collection exists and overwrite is True, the collection is recreated.
    
#     Args:
#         docs: List of documents to store in Qdrant
#         embedding: Embedding model to use
#         url: URL of the Qdrant server
#         collection_name: Name of the collection to create or overwrite
#         prefer_grpc: Whether to use gRPC for communication
#         overwrite: Whether to overwrite the collection if it exists
        
#     Returns:
#         QdrantVectorStore: Initialized Qdrant vector store
#     """
#     # Initialize Qdrant client
#     client = QdrantClient(url=url, prefer_grpc=prefer_grpc)
    
#     # Check if collection exists
#     collections = client.get_collections().collections
#     collection_exists = any(collection.name == collection_name for collection in collections)
    
#     # If collection exists and overwrite is True, delete it
#     if collection_exists and overwrite:
#         client.delete_collection(collection_name=collection_name)
#         print(f"Collection '{collection_name}' has been deleted and will be recreated.")
    
#     # Create new QdrantVectorStore from documents
#     qdrant = QdrantVectorStore.from_documents(
#         docs,
#         embedding=embedding,
#         url=url,
#         prefer_grpc=prefer_grpc,
#         collection_name=collection_name,
#     )
    
#     print(f"Successfully initialized Qdrant collection '{collection_name}'.")
#     return qdrant


# #####################################################################################
# ####                      5.     Get existing Qdrant store                       ####
# #####################################################################################

# def get_existing_qdrant_store(
#     embedding: Embeddings,
#     url: str = "http://localhost:6333",
#     collection_name: str = "my_documents",
#     api_key: Optional[str] = None,
#     prefer_grpc: bool = True,
#     content_payload_key: str = "page_content"
# ) -> QdrantVectorStore:
#     """Connect to an existing Qdrant collection without loading new documents.
    
#     Args:
#         embedding: Embedding model to use for searches
#         url: URL of the Qdrant server
#         collection_name: Name of the existing collection
#         api_key: API key for Qdrant cloud (optional)
#         prefer_grpc: Whether to use gRPC for communication
#         content_payload_key: The metadata key that contains the document content
        
#     Returns:
#         QdrantVectorStore: Connected Qdrant vector store
        
#     Raises:
#         ValueError: If the collection doesn't exist
#     """
#     # Initialize Qdrant client to check if collection exists
#     client = QdrantClient(url=url, prefer_grpc=prefer_grpc, api_key=api_key)
    
#     # Check if collection exists
#     collections = client.get_collections().collections
#     collection_exists = any(collection.name == collection_name for collection in collections)
    
#     if not collection_exists:
#         raise ValueError(f"Collection '{collection_name}' does not exist. Use init_qdrant to create it.")
    
#     # Connect to existing Qdrant collection
#     qdrant = QdrantVectorStore.from_existing_collection(
#         embedding=embedding,
#         collection_name=collection_name,
#         url=url,
#         prefer_grpc=prefer_grpc,
#         api_key=api_key,
#         content_payload_key=content_payload_key
#     )
    
#     print(f"Successfully connected to existing Qdrant collection '{collection_name}'.")
#     return qdrant

# #####################################################################################
# ####                      6.      Init Qdrant from text                          ####
# #####################################################################################

# def init_qdrant_from_texts(
#     texts: List[str],
#     embedding: Embeddings,
#     url: str = "http://localhost:6333",
#     collection_name: str = "my_texts",
#     api_key: Optional[str] = None,
#     prefer_grpc: bool = True,
#     metadatas: Optional[List[dict]] = None,
#     overwrite: bool = True
# ) -> QdrantVectorStore:
#     """Initialize a Qdrant vector store with the given text strings.
    
#     Args:
#         texts: List of text strings to store
#         embedding: Embedding model to use
#         url: URL of the Qdrant server
#         collection_name: Name of the collection
#         api_key: API key for Qdrant cloud (optional)
#         prefer_grpc: Whether to use gRPC for communication
#         metadatas: Optional metadata for each text string
#         overwrite: Whether to overwrite the collection if it exists
        
#     Returns:
#         QdrantVectorStore: Initialized Qdrant vector store
#     """
#     # Initialize Qdrant client
#     client = QdrantClient(url=url, prefer_grpc=prefer_grpc, api_key=api_key)
    
#     # Check if collection exists
#     collections = client.get_collections().collections
#     collection_exists = any(collection.name == collection_name for collection in collections)
    
#     # If collection exists and overwrite is True, delete it
#     if collection_exists and overwrite:
#         client.delete_collection(collection_name=collection_name)
#         print(f"Collection '{collection_name}' has been deleted and will be recreated.")
    
#     # Create new QdrantVectorStore from texts
#     qdrant = QdrantVectorStore.from_texts(
#         texts=texts,
#         embedding=embedding,
#         url=url,
#         prefer_grpc=prefer_grpc,
#         collection_name=collection_name,
#         metadatas=metadatas,
#         api_key=api_key
#     )
    
#     print(f"Successfully initialized Qdrant collection '{collection_name}' from text strings.")
#     return qdrant

 

# #####################################################################################
# ####                       7.       Dense vector search                          ####
# #####################################################################################
# def dense_vector_search(
#     query: str,
#     embedding: Embeddings,
#     url: str = "http://localhost:6333",
#     collection_name: str = "my_documents",
#     k: int = 4,
#     api_key: Optional[str] = None,
#     filter_condition: Optional[dict] = None
# ) -> List[Document]:
#     """Perform dense vector search on an existing Qdrant collection.
    
#     Args:
#         query: The search query
#         embedding: Dense embedding model to use
#         url: URL of the Qdrant server
#         collection_name: Name of the collection
#         k: Number of documents to return
#         api_key: API key for Qdrant cloud (optional)
#         filter_condition: Optional filter condition to apply
        
#     Returns:
#         List[Document]: Retrieved documents
#     """
#     # Connect to existing Qdrant collection
#     qdrant = get_existing_qdrant_store(
#         embedding=embedding,
#         url=url,
#         collection_name=collection_name,
#         api_key=api_key
#     )
    
#     # Perform similarity search
#     found_docs = qdrant.similarity_search(
#         query=query, 
#         k=k,
#         filter=filter_condition
#     )
    
#     return found_docs


# #####################################################################################
# ####                       8.    Sparse vector search                            ####
# #####################################################################################
# def sparse_vector_search(
#     query: str,
#     url: str = "http://localhost:6333",
#     collection_name: str = "my_documents",
#     sparse_model_name: str = "Qdrant/BM25",
#     k: int = 4,
#     api_key: Optional[str] = None,
#     filter_condition: Optional[dict] = None
# ) -> List[Document]:
#     """Perform sparse vector search on an existing Qdrant collection.
    
#     Args:
#         query: The search query
#         url: URL of the Qdrant server
#         collection_name: Name of the collection
#         sparse_model_name: Name of the sparse embedding model
#         k: Number of documents to return
#         api_key: API key for Qdrant cloud (optional)
#         filter_condition: Optional filter condition to apply
        
#     Returns:
#         List[Document]: Retrieved documents
#     """
#     # Initialize sparse embeddings
#     sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)
    
#     # Connect to existing Qdrant collection
#     qdrant = QdrantVectorStore.from_existing_collection(
#         sparse_embedding=sparse_embeddings,
#         collection_name=collection_name,
#         url=url,
#         api_key=api_key,
#         retrieval_mode=RetrievalMode.SPARSE
#     )
    
#     # Perform similarity search
#     found_docs = qdrant.similarity_search(
#         query=query, 
#         k=k,
#         filter=filter_condition
#     )
    
#     return found_docs

# #####################################################################################
# ####                      9.      Hybrid vector search                           ####
# #####################################################################################
# def hybrid_vector_search(
#     query: str,
#     embedding: Embeddings,
#     url: str = "http://localhost:6333",
#     collection_name: str = "my_documents",
#     sparse_model_name: str = "Qdrant/BM25",
#     k: int = 4,
#     api_key: Optional[str] = None,
#     filter_condition: Optional[dict] = None
# ) -> List[Document]:
#     """Perform hybrid vector search on an existing Qdrant collection using both dense and sparse vectors.
    
#     Args:
#         query: The search query
#         embedding: Dense embedding model to use
#         url: URL of the Qdrant server
#         collection_name: Name of the collection
#         sparse_model_name: Name of the sparse embedding model
#         k: Number of documents to return
#         api_key: API key for Qdrant cloud (optional)
#         filter_condition: Optional filter condition to apply
        
#     Returns:
#         List[Document]: Retrieved documents
#     """
#     # Initialize sparse embeddings
#     sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)
    
#     # Connect to existing Qdrant collection
#     qdrant = QdrantVectorStore.from_existing_collection(
#         embedding=embedding,
#         sparse_embedding=sparse_embeddings,
#         collection_name=collection_name,
#         url=url,
#         api_key=api_key,
#         retrieval_mode=RetrievalMode.HYBRID
#     )
    
#     # Perform similarity search
#     found_docs = qdrant.similarity_search(
#         query=query, 
#         k=k,
#         filter=filter_condition
#     )
    
#     return found_docs


# #####################################################################################
# ##     10.  Convenience function to create a search function based on the mode     ##
# #####################################################################################
# #
# def create_vector_search(mode: str = "dense") -> Callable:
#     """Create a vector search function based on the specified mode.
    
#     Args:
#         mode: The search mode to use ("dense", "sparse", or "hybrid")
        
#     Returns:
#         function: The appropriate vector search function
        
#     Raises:
#         ValueError: If an invalid mode is specified
#     """
#     mode = mode.lower()
#     if mode == "dense":
#         return dense_vector_search
#     elif mode == "sparse":
#         return sparse_vector_search
#     elif mode == "hybrid":
#         return hybrid_vector_search
#     else:
#         raise ValueError(f"Invalid search mode: {mode}. Must be 'dense', 'sparse', or 'hybrid'.")
    

