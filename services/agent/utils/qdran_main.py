from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings


import os
from loguru import logger

from typing import Optional

def main(
    url: str,
    collection_name: str,
    prefer_grpc: bool,
    file_path: str,
    hf_model: str,
    api_key: Optional[str],
    content_payload_key: Optional[str],
    huggingface_api_key: Optional[str],
    deepseek_api_key: str,
    deepseek_url: str,
    deepseek_model: str,
):
    """
    Main function for ingest services:

    Args:
        url: str: URL of the Qdrant server
        collection_name: str: Name of the collection to ingest
        prefer_grpc: bool: Prefer gRPC over HTTP
        file_path: str: Path to the file containing the data to ingest
        hf_model: str: Hugging Face model to use for embeddings
        api_key: str: API key for the Qdrant server
        content_payload_key: str: Key in the JSON payload containing the content to ingest
        huggingface_api_key: str: API key for Hugging Face

    Returns:
        None
    
    """



if __name__ == "__main__":
    
    from config import config
    
    main(
        url=config.url,
        collection_name=config.collection_name,
        prefer_grpc=config.prefer_grpc,
        file_path=config.file_path,
        hf_model=config.hf_model,
        api_key=config.api_key,
        content_payload_key=config.content_payload_key,
        huggingface_api_key=config.huggingface_api_key,
        deepseek_api_key=config.deepseek_api_key
    )

    pass