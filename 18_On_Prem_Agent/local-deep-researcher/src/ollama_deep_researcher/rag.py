"""RAG (Retrieval-Augmented Generation) module using QDrant vector store."""

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

# Constants
DEFAULT_COLLECTION_NAME = "DnD_Documents"
DEFAULT_QDRANT_URL = "127.0.0.1:6334"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_DATA_PATH = "./data/data"


def get_qdrant_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    recreate_collection: bool = False,
) -> QdrantVectorStore:
    """Initialize or get existing QDrant vector store.
    
    Args:
        collection_name: Name of the QDrant collection
        qdrant_url: QDrant server URL (host:port)
        embedding_model: Name of the Ollama embedding model to use
        recreate_collection: If True, recreate the collection (deletes existing data)
        
    Returns:
        QdrantVectorStore instance
    """
    logger.info(f"Initializing QDrant vector store: collection={collection_name}, url={qdrant_url}")
    
    from qdrant_client import QdrantClient
    
    # Parse URL - handle both "host:port" and "http://host:port" formats
    if "://" in qdrant_url:
        # Full URL format
        url_parts = qdrant_url.split("://")[1]
        if ":" in url_parts:
            host, port = url_parts.split(":")
            port = int(port)
        else:
            host = url_parts
            port = 6333
    else:
        # host:port format
        if ":" in qdrant_url:
            host, port = qdrant_url.split(":")
            port = int(port)
        else:
            host = qdrant_url
            port = 6333
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Create Qdrant client
    client = QdrantClient(host=host, port=port, prefer_grpc=True)
    
    # Handle collection recreation if requested
    if recreate_collection:
        logger.info(f"Recreating collection: {collection_name}")
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            logger.debug(f"Collection may not exist: {e}")
    
    # Try to get existing collection or create new one
    try:
        # Check if collection exists
        collection_info = client.get_collection(collection_name)
        logger.info(f"Found existing collection: {collection_name} with {collection_info.points_count} points")
        # Use from_existing_collection to connect to existing collection
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            embedding=embeddings,
            host=host,
            port=port,
            prefer_grpc=True,
        )
    except Exception:
        # Collection doesn't exist, create it
        logger.info(f"Collection does not exist, creating new one: {collection_name}")
        vector_store = QdrantVectorStore.from_texts(
            texts=["Initial document"],  # Temporary document to create collection
            embedding=embeddings,
            host=host,
            port=port,
            prefer_grpc=True,
            collection_name=collection_name,
        )
        # Delete the temporary document
        try:
            results, _ = vector_store.client.scroll(collection_name, limit=1)
            if results:
                vector_store.delete([results[0].id])
        except:
            pass
    
    return vector_store


def load_documents_to_qdrant(
    data_path: str = DEFAULT_DATA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    recreate_collection: bool = False,
) -> QdrantVectorStore:
    """Load documents from data folder into QDrant vector store.
    
    Args:
        data_path: Path to the data directory containing JSON files
        collection_name: Name of the QDrant collection
        qdrant_url: QDrant server URL (host:port)
        embedding_model: Name of the Ollama embedding model to use
        recreate_collection: If True, recreate the collection before loading
        
    Returns:
        QdrantVectorStore instance with loaded documents
    """
    logger.info(f"Loading documents from {data_path} into QDrant")
    
    # Check if data path exists
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        logger.warning(f"Data path does not exist: {data_path}. Creating empty vector store.")
        return get_qdrant_vector_store(
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
            recreate_collection=recreate_collection,
        )
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Load documents
    try:
        json_loader = DirectoryLoader(
            path=str(data_path_obj),
            glob="**/*.json",
            loader_cls=JSONLoader,
            loader_kwargs={"jq_schema": "..", "text_content": False}
        )
        json_documents = json_loader.load()
        logger.info(f"Loaded {len(json_documents)} documents from {data_path}")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        logger.info("Creating empty vector store")
        return get_qdrant_vector_store(
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
            recreate_collection=recreate_collection,
        )
    
    if len(json_documents) == 0:
        logger.warning(f"No documents found in {data_path}")
        return get_qdrant_vector_store(
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
            recreate_collection=recreate_collection,
        )
    
    # Parse URL for collection deletion if needed
    if recreate_collection:
        logger.info("Recreating collection before loading documents")
        try:
            from qdrant_client import QdrantClient
            # Parse URL
            if "://" in qdrant_url:
                url_parts = qdrant_url.split("://")[1]
                if ":" in url_parts:
                    host, port = url_parts.split(":")
                    port = int(port)
                else:
                    host = url_parts
                    port = 6333
            else:
                if ":" in qdrant_url:
                    host, port = qdrant_url.split(":")
                    port = int(port)
                else:
                    host = qdrant_url
                    port = 6333
            client = QdrantClient(host=host, port=port, prefer_grpc=True)
            try:
                client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except:
                pass
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
    
    # Parse URL - handle both "host:port" and "http://host:port" formats
    if "://" in qdrant_url:
        # Full URL format
        url_parts = qdrant_url.split("://")[1]
        if ":" in url_parts:
            host, port = url_parts.split(":")
            port = int(port)
        else:
            host = url_parts
            port = 6333
    else:
        # host:port format
        if ":" in qdrant_url:
            host, port = qdrant_url.split(":")
            port = int(port)
        else:
            host = qdrant_url
            port = 6333
    
    # Load documents into QDrant
    logger.info(f"Loading {len(json_documents)} documents into QDrant (this may take a while)...")
    qdrant = QdrantVectorStore.from_documents(
        json_documents,
        embeddings,
        host=host,
        port=port,
        prefer_grpc=True,
        collection_name=collection_name,
    )
    logger.info(f"Successfully loaded documents into QDrant collection: {collection_name}")
    
    return qdrant


def query_rag(
    query: str,
    vector_store: QdrantVectorStore,
    k: int = 5,
) -> List[Document]:
    """Query the RAG vector store for relevant documents.
    
    Args:
        query: Search query string
        vector_store: QDrantVectorStore instance
        k: Number of documents to retrieve
        
    Returns:
        List of relevant Document objects
    """
    logger.info(f"Querying RAG with query: '{query}', k={k}")
    
    try:
        results = vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} documents from RAG")
        return results
    except Exception as e:
        logger.error(f"Error querying RAG: {e}")
        return []


def format_rag_results(documents: List[Document]) -> str:
    """Format RAG search results into a readable string.
    
    Args:
        documents: List of Document objects from RAG query
        
    Returns:
        Formatted string with document content
    """
    if not documents:
        return ""
    
    formatted_text = "=== RAG Context (from Local Knowledge Base) ===\n\n"
    for i, doc in enumerate(documents, 1):
        formatted_text += f"Document {i}:\n"
        formatted_text += f"{doc.page_content}\n"
        if doc.metadata:
            formatted_text += f"Metadata: {doc.metadata}\n"
        formatted_text += "\n---\n\n"
    
    return formatted_text.strip()

