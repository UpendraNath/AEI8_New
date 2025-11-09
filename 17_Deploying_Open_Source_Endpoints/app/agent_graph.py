"""Agent graph implementation using Together AI endpoints.

This module implements a LangGraph agent that can use RAG and other tools
to answer user queries using Together AI models.
"""
from __future__ import annotations

from typing import Annotated, List, TypedDict, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_together import ChatTogether


class AgentState(TypedDict):
    """State schema for agent graphs, storing a message list with add_messages."""
    messages: Annotated[List, add_messages]


@tool
def retrieve_information(
    query: str
) -> str:
    """Search and retrieve information from loaded PDF documents using RAG.
    
    This tool searches through PDF documents loaded from the data directory and answers questions 
    based on the document content. You MUST use this tool for ANY question about information, 
    topics, usage, examples, summaries, or content that might be in the documents.
    
    CRITICAL: Always use this tool when users ask about:
    - Any information, topics, or content
    - Usage patterns, how something is used
    - Examples, summaries, main points
    - Questions starting with "What", "How", "Can you find", etc.
    
    Never say you don't have information without first using this tool!
    
    Args:
        query: The question or query to search for (e.g., "AI usage", "What are the main points?", "How is ChatGPT used?")
        
    Returns:
        The answer retrieved from the documents
    """
    from app.rag import get_rag_response
    import os
    
    model_endpoint = os.environ.get("TOGETHER_MODEL_ENDPOINT", "openai/gpt-oss-20b")
    return get_rag_response(query, model_endpoint)


def get_tool_belt():
    """Return the list of tools available to agents."""
    return [retrieve_information]


def build_model_with_tools(model, model_endpoint: str):
    """Return a model instance bound to the tool belt."""
    return model.bind_tools(get_tool_belt())


def call_model(state: Dict[str, Any], model, model_endpoint: str) -> Dict[str, Any]:
    """Invoke the model with the accumulated messages and append its response."""
    model_with_tools = build_model_with_tools(model, model_endpoint)
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_end(state: Dict[str, Any]):
    """Decide whether to execute tools or end."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return END


def build_agent_graph(model_endpoint: str = "openai/gpt-oss-20b"):
    """Build an agent graph that can use RAG and other tools.
    
    Args:
        model_endpoint: The Together AI model endpoint identifier
        
    Returns:
        A compiled LangGraph agent
    """
    from langchain_core.messages import SystemMessage
    
    # System prompt to encourage tool usage
    system_prompt = (
        "You are a helpful AI assistant with access to a document retrieval tool called 'retrieve_information'. "
        "You have PDF documents loaded in the system that contain information about various topics. "
        "CRITICAL RULE: For ANY user question, you MUST ALWAYS use the retrieve_information tool FIRST. "
        "Do NOT answer from memory or say you don't have information - ALWAYS call the tool first. "
        "The tool will search the documents and return relevant information. "
        "Only if the tool explicitly returns 'I don't know' should you indicate that no information was found. "
        "Examples of when to use the tool: 'AI usage', 'What information can you find', 'topics', 'examples', etc."
    )
    
    model = ChatTogether(
        model=model_endpoint,
        temperature=0.7,
    )
    
    # Create model-bound function with system prompt
    def _call_model(state: AgentState) -> Dict[str, Any]:
        """Wrapper to pass model to call_model with system prompt."""
        messages = list(state["messages"])
        # Add system prompt if not already present
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system:
            messages = [SystemMessage(content=system_prompt)] + messages
        state_with_system = {**state, "messages": messages}
        return call_model(state_with_system, model, model_endpoint)
    
    # Create tool node
    tool_node = ToolNode(get_tool_belt())
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", _call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_end,
        {"action": "action", END: END},
    )
    graph.add_edge("action", "agent")
    
    return graph.compile()


def run_agent(query: str, model_endpoint: str = "openai/gpt-oss-20b") -> str:
    """Run the agent with a query and return the final response.
    
    Args:
        query: The user's question
        model_endpoint: The Together AI model endpoint identifier
        
    Returns:
        The agent's final response
    """
    from langchain_core.messages import ToolMessage
    
    graph = build_agent_graph(model_endpoint)
    result = graph.invoke({"messages": [("user", query)]})
    
    # Extract the final message - look for the last AIMessage that doesn't have tool_calls
    messages = result.get("messages", [])
    if messages:
        # Check if tool was called - look for ToolMessage
        tool_was_called = any(isinstance(msg, ToolMessage) for msg in messages)
        
        # Go through messages in reverse to find the final response
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                # If this message has tool calls, continue looking
                if hasattr(message, "tool_calls") and message.tool_calls:
                    continue
                # Return the content, but clean it up if it has analysis text
                content = message.content
                if content:
                    # Clean up content - remove analysis prefixes
                    content_str = str(content)
                    
                    # Remove "analysis" prefix and extract final answer
                    if "assistantfinal" in content_str.lower():
                        # Split on assistantfinal and take the part after it
                        parts = content_str.split("assistantfinal", 1)
                        if len(parts) > 1:
                            return parts[1].strip()
                    
                    # Remove "analysis" prefix if present
                    if content_str.lower().startswith("analysis"):
                        # Try to find where the actual response starts
                        # Look for common markers after "analysis"
                        remaining = content_str[len("analysis"):].strip()
                        for marker in ["assistantfinal", "final", "answer:", "response:"]:
                            if marker.lower() in remaining.lower():
                                marker_idx = remaining.lower().find(marker.lower())
                                if marker_idx >= 0:
                                    return remaining[marker_idx + len(marker):].strip()
                        # If no marker found, return everything after "analysis"
                        return remaining
                    
                    # If tool wasn't called and response says no info, try calling tool directly
                    if not tool_was_called and ("couldn't find" in content_str.lower() or "don't have" in content_str.lower() or "no information" in content_str.lower()):
                        # Force tool call as fallback
                        from app.rag import get_rag_response
                        tool_response = get_rag_response(query, model_endpoint)
                        if tool_response and "don't know" not in tool_response.lower():
                            return tool_response
                    
                    return content_str
    
    return "No response generated."

