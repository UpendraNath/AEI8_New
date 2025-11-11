"""Script to initialize QDrant vector store with documents from data folder.

This script loads documents from the data folder and indexes them in QDrant.
Run this script once before using the RAG functionality.

Usage:
    python -m ollama_deep_researcher.init_rag
    or
    python scripts/init_rag.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ollama_deep_researcher.rag import load_documents_to_qdrant
from ollama_deep_researcher.configuration import Configuration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize QDrant vector store with documents."""
    logger.info("Starting QDrant vector store initialization...")
    
    # Get configuration (can be overridden with environment variables)
    config = Configuration()
    
    logger.info(f"Configuration:")
    logger.info(f"  - QDrant URL: {config.qdrant_url}")
    logger.info(f"  - Collection Name: {config.qdrant_collection_name}")
    logger.info(f"  - Embedding Model: {config.embedding_model}")
    logger.info(f"  - Data Path: {config.rag_data_path}")
    
    # Load documents into QDrant
    try:
        vector_store = load_documents_to_qdrant(
            data_path=config.rag_data_path,
            collection_name=config.qdrant_collection_name,
            qdrant_url=config.qdrant_url,
            embedding_model=config.embedding_model,
            recreate_collection=False,  # Set to True to recreate collection
        )
        logger.info("✅ Successfully initialized QDrant vector store!")
        logger.info(f"Collection '{config.qdrant_collection_name}' is ready for queries.")
    except Exception as e:
        logger.error(f"❌ Error initializing QDrant vector store: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

