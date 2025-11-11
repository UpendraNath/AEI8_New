import json
import logging
import re

from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import (
    deduplicate_and_format_sources,
    tavily_search,
    format_sources,
    perplexity_search,
    duckduckgo_search,
    searxng_search,
    strip_thinking_tokens,
    get_config_value,
)
from ollama_deep_researcher.state import (
    SummaryState,
    SummaryStateInput,
    SummaryStateOutput,
)
from ollama_deep_researcher.prompts import (
    query_writer_instructions,
    summarizer_instructions,
    reflection_instructions,
    get_current_date,
    json_mode_query_instructions,
    tool_calling_query_instructions,
    json_mode_reflection_instructions,
    tool_calling_reflection_instructions,
)
from ollama_deep_researcher.lmstudio import ChatLMStudio
from ollama_deep_researcher.rag import (
    get_qdrant_vector_store,
    query_rag,
    format_rag_results,
    load_documents_to_qdrant,
)

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MAX_TOKENS_PER_SOURCE = 1000
CHARS_PER_TOKEN = 4

def generate_search_query_with_structured_output(
    configurable: Configuration,
    messages: list,
    tool_class,
    fallback_query: str,
    tool_query_field: str,
    json_query_field: str,
):
    """Helper function to generate search queries using either tool calling or JSON mode.
    
    Args:
        configurable: Configuration object
        messages: List of messages to send to LLM
        tool_class: Tool class for tool calling mode
        fallback_query: Fallback search query if extraction fails
        tool_query_field: Field name in tool args containing the query
        json_query_field: Field name in JSON response containing the query
        
    Returns:
        Dictionary with "search_query" key
    """
    try:
        if configurable.use_tool_calling:
            llm = get_llm(configurable).bind_tools([tool_class])
            result = llm.invoke(messages)

            if not result.tool_calls:
                return {"search_query": fallback_query}
            
            try:
                tool_data = result.tool_calls[0]["args"]
                search_query = tool_data.get(tool_query_field)
                return {"search_query": search_query}
            except (IndexError, KeyError):
                return {"search_query": fallback_query}
        
        else:
            # Use JSON mode
            llm = get_llm(configurable)
            result = llm.invoke(messages)
            logger.info(f"LLM result type: {type(result)}, result: {result}")
            
            # Extract content - invoke() returns AIMessage directly
            content = None
            
            # Try direct content access (AIMessage from invoke)
            if hasattr(result, 'content'):
                content = result.content
                logger.info(f"Accessed content directly from result (AIMessage), content type: {type(content)}")
            # Try ChatResult structure (generations -> message -> content) as fallback
            elif hasattr(result, 'generations') and result.generations:
                generation = result.generations[0]
                if hasattr(generation, 'message') and generation.message:
                    if hasattr(generation.message, 'content'):
                        content = generation.message.content
                        logger.info("Accessed content from ChatResult -> ChatGeneration -> AIMessage")
            
            # Handle missing or empty content
            if content is None:
                logger.error("Could not extract content from LLM result, using fallback query")
                logger.error(f"Result type: {type(result)}")
                logger.error(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                if hasattr(result, 'generations'):
                    logger.error(f"Generations: {result.generations}")
                return {"search_query": fallback_query}
            
            # Handle content that might be a list (AIMessage content can be str | list[str | dict])
            if isinstance(content, list):
                logger.info(f"Content is a list, extracting text from first element")
                if len(content) > 0:
                    # Extract text from content blocks
                    if isinstance(content[0], dict):
                        content = content[0].get('text', '') or content[0].get('content', '')
                    else:
                        content = str(content[0])
                else:
                    content = ""
            
            # Convert to string if not already
            if not isinstance(content, str):
                content = str(content)
            
            # Handle empty string content
            if not content or not content.strip():
                logger.warning(f"LLM message content is empty or whitespace only. Content: {repr(content)}, using fallback query")
                return {"search_query": fallback_query}
            
            logger.info(f"Extracted content type: {type(content)}, length: {len(content)}")
            logger.info(f"Raw LLM content (first 500 chars): {content[:500] if len(content) > 500 else content}")

            # Strip thinking tokens first if configured
            if configurable.strip_thinking_tokens:
                content = strip_thinking_tokens(str(content))
                logger.debug(f"Content after stripping thinking tokens: {content[:500] if len(content) > 500 else content}")

            # Try to parse JSON
            try:
                # Try to extract JSON from content if it's wrapped in text
                json_start = str(content).find("{")
                json_end = str(content).rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = str(content)[json_start:json_end]
                    logger.debug(f"Extracted JSON text: {json_text}")
                    parsed_json = json.loads(json_text)
                else:
                    # Try parsing the whole content
                    parsed_json = json.loads(str(content))
                
                search_query = parsed_json.get(json_query_field)
                if not search_query or (isinstance(search_query, str) and not search_query.strip()):
                    logger.warning(f"Query field '{json_query_field}' is empty or missing in JSON: {parsed_json}, using fallback")
                    return {"search_query": fallback_query}
                logger.info(f"Successfully extracted query: {search_query}")
                return {"search_query": search_query}
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse JSON from content: {str(e)}")
                logger.warning(f"Content that failed to parse: {content[:500] if len(str(content)) > 500 else content}")
                # Try to extract query from text if JSON parsing fails
                # Look for common patterns like "query": "..." or query: "..."
                patterns = [
                    rf'["\']?{json_query_field}["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                    rf'"{json_query_field}"\s*:\s*"([^"]+)"',
                ]
                for pattern in patterns:
                    match = re.search(pattern, str(content), re.IGNORECASE)
                    if match:
                        extracted_query = match.group(1).strip()
                        if extracted_query:
                            logger.info(f"Extracted query using regex pattern: {extracted_query}")
                            return {"search_query": extracted_query}
                
                logger.warning(f"Could not extract query from content, using fallback: {fallback_query}")
                return {"search_query": fallback_query}
    except Exception as e:
        # Handle connection errors and provide helpful error messages
        error_msg = str(e)
        error_type = type(e).__name__
        provider = configurable.llm_provider
        base_url = configurable.lmstudio_base_url if provider == "lmstudio" else configurable.ollama_base_url
        model = configurable.local_llm
        
        if "Connection" in error_msg or "refused" in error_msg.lower() or "10061" in error_msg:
            logger.error(
                f"Connection error: Cannot connect to {provider} at {base_url}. "
                f"Please ensure {provider} is running and accessible at this URL."
            )
            raise ConnectionError(
                f"Cannot connect to {provider} server at {base_url}. "
                f"Please ensure:\n"
                f"1. {provider.capitalize()} is installed and running\n"
                f"2. The server is accessible at {base_url}\n"
                f"3. For Ollama: Run 'ollama serve' or ensure the Ollama service is running\n"
                f"4. For LMStudio: Start the local server in LMStudio's 'Local Server' tab"
            ) from e
        elif "not found" in error_msg.lower():
            logger.error(
                f"Model '{model}' not found in {provider}: {error_type}: {error_msg}"
            )
            raise ValueError(
                f"Model '{model}' not found in {provider}. "
                f"Available models:\n"
                f"- deepseek-r1:8b (5.2 GB)\n"
                f"- gpt-oss:20b (13 GB - requires significant memory)\n"
                f"- embeddinggemma:latest (621 MB)\n\n"
                f"To fix:\n"
                f"1. Pull the model: 'ollama pull {model}'\n"
                f"2. Or use an available model by setting LOCAL_LLM environment variable\n"
                f"3. Check available models: 'ollama list'"
            ) from e
        elif "not enough space" in error_msg.lower() or "disk" in error_msg.lower() or ("space" in error_msg.lower() and "memory" not in error_msg.lower()):
            logger.error(
                f"Disk space error with {provider} model '{model}': {error_type}: {error_msg}"
            )
            raise RuntimeError(
                f"Disk space error when loading model '{model}' with {provider}. "
                f"The disk where Ollama stores models is full.\n\n"
                f"Solutions:\n"
                f"1. Free up disk space on C: drive (currently 0 GB free)\n"
                f"2. Move Ollama models to D: drive:\n"
                f"   - Set environment variable: OLLAMA_MODELS=D:\\ollama\\models\n"
                f"   - Restart Ollama service\n"
                f"3. Delete unused models: 'ollama rm <model-name>'\n"
                f"4. Clean up temporary files and other applications\n"
                f"5. Check disk space: 'Get-PSDrive -PSProvider FileSystem'"
            ) from e
        elif "memory" in error_msg.lower() or "allocation" in error_msg.lower() or "ResponseError" in error_type:
            logger.error(
                f"Memory/allocation error with {provider} model '{model}': {error_type}: {error_msg}"
            )
            raise RuntimeError(
                f"Memory allocation error when loading model '{model}' with {provider}. "
                f"This usually means:\n"
                f"1. The model is too large for available RAM/VRAM\n"
                f"2. Multiple models are loaded simultaneously\n"
                f"3. Insufficient system memory\n"
                f"4. Disk space may be full (check with 'Get-PSDrive')\n\n"
                f"Solutions:\n"
                f"- Try a smaller model (e.g., 'deepseek-r1:8b' instead of 'gpt-oss:20b')\n"
                f"- Free up memory by closing other applications\n"
                f"- Unload other models: 'ollama ps' to see loaded models\n"
                f"- Check available models: 'ollama list'\n"
                f"- Check disk space: 'Get-PSDrive -PSProvider FileSystem'\n"
                f"- For Ollama, try: 'ollama run {model}' to test if the model loads"
            ) from e
        else:
            # Re-raise other errors as-is
            logger.error(f"Unexpected error with {provider} model '{model}': {error_type}: {error_msg}")
            raise

def get_llm(configurable: Configuration):
    """Helper function to initialize LLM based on configuration.

    Uses JSON mode if use_tool_calling is False, otherwise regular mode for tool calling.

    Args:
        configurable: Configuration object containing LLM settings

    Returns:
        Configured LLM instance
        
    Raises:
        ValueError: If the LLM provider is not supported or configuration is invalid
    """
    # Normalize base URLs
    if configurable.llm_provider == "lmstudio":
        # Ensure LMStudio base URL ends with /v1
        base_url = configurable.lmstudio_base_url.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = base_url + '/v1'
        
        logger.info(f"Initializing LMStudio LLM with base_url={base_url}, model={configurable.local_llm}")
        
        if configurable.use_tool_calling:
            return ChatLMStudio(
                base_url=base_url,
                model=configurable.local_llm,
                temperature=0,
            )
        else:
            return ChatLMStudio(
                base_url=base_url,
                model=configurable.local_llm,
                temperature=0,
                format="json",
            )
    else:  # Default to Ollama
        # Ensure Ollama base URL ends with /
        base_url = configurable.ollama_base_url.rstrip('/') + '/'
        
        logger.info(f"Initializing Ollama LLM with base_url={base_url}, model={configurable.local_llm}")
        
        if configurable.use_tool_calling:
            return ChatOllama(
                base_url=base_url,
                model=configurable.local_llm,
                temperature=0,
            )
        else:
            return ChatOllama(
                base_url=base_url,
                model=configurable.local_llm,
                temperature=0,
                format="json",
            )

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the research topic.

    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Supports both LMStudio and Ollama as LLM providers.

    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date, research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)

    @tool
    class Query(BaseModel):
        """
        This tool is used to generate a query for web search.
        """

        query: str = Field(description="The actual search query string")
        rationale: str = Field(
            description="Brief explanation of why this query is relevant"
        )

    messages = [
        SystemMessage(
            content=formatted_prompt + (
                tool_calling_query_instructions if configurable.use_tool_calling 
                else json_mode_query_instructions
            )
        ),
        HumanMessage(content="Generate a query for web search:"),
    ]
    
    # Log the prompt for debugging
    logger.info(f"Generating query for research topic: {state.research_topic}")
    logger.debug(f"System message content: {messages[0].content[:500]}...")
    logger.debug(f"Using tool calling: {configurable.use_tool_calling}")

    return generate_search_query_with_structured_output(
        configurable=configurable,
        messages=messages,
        tool_class=Query,
        fallback_query=f"Tell me more about {state.research_topic}",
        tool_query_field="query",
        json_query_field="query",
    )


def web_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs web research using the generated search query.

    Executes a web search using the configured search API (tavily, perplexity,
    duckduckgo, or searxng) and formats the results for further processing.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Get the search API
    search_api = get_config_value(configurable.search_api)

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(
            state.search_query,
            fetch_full_page=configurable.fetch_full_page,
            max_results=1,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == "perplexity":
        search_results = perplexity_search(
            state.search_query, state.research_loop_count
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == "searxng":
        search_results = searxng_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str],
    }


def query_rag_knowledge_base(state: SummaryState, config: RunnableConfig):
    """LangGraph node that queries the local RAG knowledge base using QDrant.
    
    Retrieves relevant documents from the local knowledge base using the research topic
    or search query as the query string. This provides additional context from the
    pre-loaded documents in QDrant.
    
    Args:
        state: Current graph state containing the research topic and search query
        config: Configuration for the runnable, including RAG settings
        
    Returns:
        Dictionary with state update, including rag_context key containing formatted RAG results
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Skip RAG if disabled
    if not configurable.use_rag:
        logger.info("RAG is disabled, skipping RAG query")
        return {"rag_context": ""}
    
    # Use research topic or search query as the query string
    query_string = state.search_query or state.research_topic
    
    if not query_string:
        logger.warning("No query string available for RAG, skipping")
        return {"rag_context": ""}
    
    logger.info(f"Querying RAG knowledge base with: '{query_string}'")
    
    try:
        # Initialize QDrant vector store
        vector_store = get_qdrant_vector_store(
            collection_name=configurable.qdrant_collection_name,
            qdrant_url=configurable.qdrant_url,
            embedding_model=configurable.embedding_model,
        )
        
        # Query RAG
        rag_documents = query_rag(
            query=query_string,
            vector_store=vector_store,
            k=configurable.rag_k,
        )
        
        # Format results
        rag_context = format_rag_results(rag_documents)
        
        if rag_context:
            logger.info(f"Retrieved {len(rag_documents)} documents from RAG")
        else:
            logger.warning("No documents retrieved from RAG")
        
        return {"rag_context": rag_context}
        
    except Exception as e:
        logger.error(f"Error querying RAG: {e}", exc_info=True)
        # Don't fail the entire flow if RAG fails
        return {"rag_context": ""}


def summarize_sources(state: SummaryState, config: RunnableConfig):
    """LangGraph node that summarizes web research results.

    Uses an LLM to create or update a running summary based on the newest web research
    results, integrating them with any existing summary.

    Args:
        state: Current graph state containing research topic, running summary,
              and web research results
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including running_summary key containing the updated summary
    """

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1] if state.web_research_results else ""
    
    # Include RAG context if available
    rag_context_section = ""
    if state.rag_context:
        rag_context_section = f"<Local Knowledge Base Context> \n {state.rag_context} \n <Local Knowledge Base Context>\n\n"

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"{rag_context_section}"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {most_recent_web_research} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"{rag_context_section}"
            f"<Context> \n {most_recent_web_research} \n <Context>"
            f"Create a Summary using the Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)

    # For summarization, we don't need structured output, so always use regular mode
    # Normalize base URLs
    if configurable.llm_provider == "lmstudio":
        base_url = configurable.lmstudio_base_url.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = base_url + '/v1'
        llm = ChatLMStudio(
            base_url=base_url,
            model=configurable.local_llm,
            temperature=0,
        )
    else:  # Default to Ollama
        base_url = configurable.ollama_base_url.rstrip('/') + '/'
        llm = ChatOllama(
            base_url=base_url,
            model=configurable.local_llm,
            temperature=0,
        )

    try:
        result = llm.invoke(
            [
                SystemMessage(content=summarizer_instructions),
                HumanMessage(content=human_message_content),
            ]
        )
        
        # Extract content from AIMessage (same logic as generate_search_query_with_structured_output)
        content = None
        
        # Try direct content access (AIMessage from invoke)
        if hasattr(result, 'content'):
            content = result.content
            logger.debug("Accessed content directly from result (AIMessage) in summarize_sources")
        # Try ChatResult structure as fallback
        elif hasattr(result, 'generations') and result.generations:
            generation = result.generations[0]
            if hasattr(generation, 'message') and generation.message:
                if hasattr(generation.message, 'content'):
                    content = generation.message.content
                    logger.debug("Accessed content from ChatResult in summarize_sources")
        
        # Handle missing or empty content
        if content is None:
            logger.error("Could not extract content from LLM result in summarize_sources")
            raise ValueError("LLM returned no content for summarization")
        
        # Handle content that might be a list
        if isinstance(content, list):
            logger.debug("Content is a list in summarize_sources, extracting text")
            if len(content) > 0:
                if isinstance(content[0], dict):
                    content = content[0].get('text', '') or content[0].get('content', '')
                else:
                    content = str(content[0])
            else:
                content = ""
        
        # Convert to string if not already
        if not isinstance(content, str):
            content = str(content)
        
        # Handle empty string content
        if not content or not content.strip():
            logger.warning(f"LLM message content is empty in summarize_sources. Content: {repr(content)}")
            raise ValueError("LLM returned empty content for summarization")
        
        logger.info(f"Extracted summary content, length: {len(content)}")
        
    except Exception as e:
        # Handle connection errors and provide helpful error messages
        error_msg = str(e)
        error_type = type(e).__name__
        provider = configurable.llm_provider
        base_url = configurable.lmstudio_base_url if provider == "lmstudio" else configurable.ollama_base_url
        model = configurable.local_llm
        
        if "Connection" in error_msg or "refused" in error_msg.lower() or "10061" in error_msg:
            logger.error(
                f"Connection error: Cannot connect to {provider} at {base_url}. "
                f"Please ensure {provider} is running and accessible at this URL."
            )
            raise ConnectionError(
                f"Cannot connect to {provider} server at {base_url}. "
                f"Please ensure:\n"
                f"1. {provider.capitalize()} is installed and running\n"
                f"2. The server is accessible at {base_url}\n"
                f"3. For Ollama: Run 'ollama serve' or ensure the Ollama service is running\n"
                f"4. For LMStudio: Start the local server in LMStudio's 'Local Server' tab"
            ) from e
        elif "not found" in error_msg.lower():
            logger.error(
                f"Model '{model}' not found in {provider}: {error_type}: {error_msg}"
            )
            raise ValueError(
                f"Model '{model}' not found in {provider}. "
                f"Available models:\n"
                f"- deepseek-r1:8b (5.2 GB)\n"
                f"- gpt-oss:20b (13 GB - requires significant memory)\n"
                f"- embeddinggemma:latest (621 MB)\n\n"
                f"To fix:\n"
                f"1. Pull the model: 'ollama pull {model}'\n"
                f"2. Or use an available model by setting LOCAL_LLM environment variable\n"
                f"3. Check available models: 'ollama list'"
            ) from e
        elif "not enough space" in error_msg.lower() or "disk" in error_msg.lower() or ("space" in error_msg.lower() and "memory" not in error_msg.lower()):
            logger.error(
                f"Disk space error with {provider} model '{model}': {error_type}: {error_msg}"
            )
            raise RuntimeError(
                f"Disk space error when loading model '{model}' with {provider}. "
                f"The disk where Ollama stores models is full.\n\n"
                f"Solutions:\n"
                f"1. Free up disk space on C: drive (currently 0 GB free)\n"
                f"2. Move Ollama models to D: drive:\n"
                f"   - Set environment variable: OLLAMA_MODELS=D:\\ollama\\models\n"
                f"   - Restart Ollama service\n"
                f"3. Delete unused models: 'ollama rm <model-name>'\n"
                f"4. Clean up temporary files and other applications\n"
                f"5. Check disk space: 'Get-PSDrive -PSProvider FileSystem'"
            ) from e
        elif "memory" in error_msg.lower() or "allocation" in error_msg.lower() or "ResponseError" in error_type:
            logger.error(
                f"Memory/allocation error with {provider} model '{model}': {error_type}: {error_msg}"
            )
            raise RuntimeError(
                f"Memory allocation error when loading model '{model}' with {provider}. "
                f"This usually means:\n"
                f"1. The model is too large for available RAM/VRAM\n"
                f"2. Multiple models are loaded simultaneously\n"
                f"3. Insufficient system memory\n"
                f"4. Disk space may be full (check with 'Get-PSDrive')\n\n"
                f"Solutions:\n"
                f"- Try a smaller model (e.g., 'deepseek-r1:8b' instead of 'gpt-oss:20b')\n"
                f"- Free up memory by closing other applications\n"
                f"- Unload other models: 'ollama ps' to see loaded models\n"
                f"- Check available models: 'ollama list'\n"
                f"- Check disk space: 'Get-PSDrive -PSProvider FileSystem'\n"
                f"- For Ollama, try: 'ollama run {model}' to test if the model loads"
            ) from e
        else:
            # Re-raise other errors as-is
            logger.error(f"Unexpected error with {provider} model '{model}': {error_type}: {error_msg}")
            raise

    # Strip thinking tokens if configured
    running_summary = content
    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)

    return {"running_summary": running_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """LangGraph node that identifies knowledge gaps and generates follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    a new search query to address those gaps. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = reflection_instructions.format(
        research_topic=state.research_topic
    )

    @tool
    class FollowUpQuery(BaseModel):
        """
        This tool is used to generate a follow-up query to address a knowledge gap.
        """

        follow_up_query: str = Field(
            description="Write a specific question to address this gap"
        )
        knowledge_gap: str = Field(
            description="Describe what information is missing or needs clarification"
        )

    messages = [
        SystemMessage(
            content=formatted_prompt + (
                tool_calling_reflection_instructions if configurable.use_tool_calling 
                else json_mode_reflection_instructions
            )
        ),
        HumanMessage(
            content=f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:"
        ),
    ]

    return generate_search_query_with_structured_output(
        configurable=configurable,
        messages=messages,
        tool_class=FollowUpQuery,
        fallback_query=f"Tell me more about {state.research_topic}",
        tool_query_field="follow_up_query",
        json_query_field="follow_up_query",
    )


def finalize_summary(state: SummaryState):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """

    # Deduplicate sources before joining
    seen_sources = set()
    unique_sources = []

    for source in state.sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split("\n"):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    # Join the deduplicated sources
    all_sources = "\n".join(unique_sources)
    state.running_summary = (
        f"## Summary\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    )
    return {"running_summary": state.running_summary}


def route_research(
    state: SummaryState, config: RunnableConfig
) -> Literal["finalize_summary", "web_research"]:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"


# Add nodes and edges
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)
builder.add_node("generate_query", generate_query)
builder.add_node("query_rag", query_rag_knowledge_base)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "query_rag")  # Query RAG after generating query
builder.add_edge("query_rag", "web_research")  # Then do web research
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()
