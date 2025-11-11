import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"


class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="LLM Model Name",
        description="Name of the LLM model to use",
    )
    llm_provider: Literal["ollama", "lmstudio"] = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider for the LLM (Ollama or LMStudio)",
    )
    search_api: Literal["perplexity", "tavily", "duckduckgo", "searxng"] = Field(
        default="duckduckgo", title="Search API", description="Web search API to use"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    use_rag: bool = Field(
        default=True,
        title="Use RAG",
        description="Enable RAG (Retrieval-Augmented Generation) using QDrant vector store",
    )
    qdrant_url: str = Field(
        default="127.0.0.1:6334",
        title="QDrant URL",
        description="QDrant server URL (host:port)",
    )
    qdrant_collection_name: str = Field(
        default="DnD_Documents",
        title="QDrant Collection Name",
        description="Name of the QDrant collection",
    )
    embedding_model: str = Field(
        default="mxbai-embed-large",
        title="Embedding Model",
        description="Ollama embedding model name",
    )
    rag_k: int = Field(
        default=5,
        title="RAG K",
        description="Number of documents to retrieve from RAG",
    )
    rag_data_path: str = Field(
        default="./data/data",
        title="RAG Data Path",
        description="Path to the data directory containing JSON files for RAG",
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        # Pydantic will handle type conversion automatically
        values: dict[str, Any] = {}
        
        for name in cls.model_fields.keys():
            # Check environment variable first (uppercase)
            env_value = os.environ.get(name.upper())
            # Then check configurable dict
            config_value = configurable.get(name)
            
            # Use environment variable if set, otherwise use configurable value
            raw_value = env_value if env_value is not None else config_value
            
            # Only add to values if not None (let Pydantic use defaults for missing values)
            if raw_value is not None:
                values[name] = raw_value

        # Create instance - Pydantic will use defaults for any missing fields and handle type conversion
        return cls(**values)
