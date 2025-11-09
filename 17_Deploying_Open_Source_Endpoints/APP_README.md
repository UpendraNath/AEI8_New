# Agentic RAG Application with Together AI Endpoints

This application implements an Agentic RAG (Retrieval-Augmented Generation) system using LangGraph and Together AI open-source endpoints.

## Overview

The application consists of:
- **RAG System** (`app/rag.py`): Document retrieval and generation using Together AI embeddings and chat models
- **Agent Graph** (`app/agent_graph.py`): LangGraph-based agent that can use RAG tools to answer queries
- **Test Notebook** (`rag_application_test.ipynb`): Jupyter notebook for testing the application

## Setup

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Set Environment Variables**:
   ```bash
   export TOGETHER_API_KEY="your-api-key-here"
   export TOGETHER_MODEL_ENDPOINT="openai/gpt-oss-20b"  # or your dedicated endpoint
   ```

   Or set them in your notebook:
   ```python
   import os
   os.environ["TOGETHER_API_KEY"] = "your-api-key-here"
   os.environ["TOGETHER_MODEL_ENDPOINT"] = "openai/gpt-oss-20b"
   ```

## Usage

### Option 1: Using the Test Notebook

Open `rag_application_test.ipynb` in Jupyter and run the cells to test the application.

### Option 2: Using Python Scripts

**Simple RAG Query**:
```python
from app.rag import get_rag_response

query = "What is this document about?"
response = get_rag_response(query, "openai/gpt-oss-20b")
print(response)
```

**Agentic RAG with LangGraph**:
```python
from app.agent_graph import run_agent

query = "What information can you find about AI usage?"
response = run_agent(query, "openai/gpt-oss-20b")
print(response)
```

### Option 3: Using the Test Script

```bash
export TOGETHER_API_KEY="your-api-key-here"
uv run python test_rag.py
```

## Architecture

### RAG System (`app/rag.py`)

- **Document Loading**: Loads PDFs from the `data/` directory
- **Text Splitting**: Token-aware chunking using tiktoken
- **Embeddings**: Uses `TogetherEmbeddings` with `BAAI/bge-large-en-v1.5` model
- **Vector Store**: In-memory Qdrant vector database
- **Generation**: Uses `ChatTogether` with your specified model endpoint

### Agent Graph (`app/agent_graph.py`)

- **LangGraph Agent**: Implements a tool-using agent
- **RAG Tool**: Integrates the RAG system as a tool the agent can use
- **Decision Making**: Agent decides when to use tools vs. respond directly

## Configuration

- **Model Endpoint**: Set via `TOGETHER_MODEL_ENDPOINT` environment variable
  - Default: `"openai/gpt-oss-20b"` (serverless)
  - Or use your dedicated endpoint: `"your-username/openai/gpt-oss-20b-unique-id"`
  
- **Embedding Model**: Uses `BAAI/bge-large-en-v1.5` (serverless endpoint)

- **Data Directory**: Set via `RAG_DATA_DIR` environment variable
  - Default: `"data"`

## Files Structure

```
.
├── app/
│   ├── __init__.py          # Package initialization
│   ├── rag.py               # RAG implementation with Together AI
│   └── agent_graph.py       # LangGraph agent implementation
├── data/                    # PDF documents for RAG
├── rag_application_test.ipynb  # Test notebook
├── test_rag.py              # Test script
└── pyproject.toml           # Dependencies
```

## Notes

- The application uses Together AI's open-source endpoints for both chat and embeddings
- Documents are loaded from the `data/` directory (PDF files)
- The vector store is in-memory, so it's rebuilt each time the application runs
- Make sure to set your Together API key before running

## Troubleshooting

1. **Import Errors**: Make sure all dependencies are installed with `uv sync`
2. **API Key Issues**: Verify your `TOGETHER_API_KEY` is set correctly
3. **Model Endpoint**: Ensure your endpoint identifier is correct (check Together AI dashboard)
4. **No Documents**: Make sure PDF files are in the `data/` directory

