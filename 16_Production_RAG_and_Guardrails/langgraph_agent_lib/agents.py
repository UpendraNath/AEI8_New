"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os
import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


def create_guardrails_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    input_guard=None,
    output_guard=None,
    strict_mode: bool = True
):
    """Create a LangGraph agent with Guardrails validation.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        input_guard: Guardrails Guard for input validation
        output_guard: Guardrails Guard for output validation
        strict_mode: If True, raises exceptions on validation failure
        
    Returns:
        Compiled LangGraph agent with guardrails
    """
    from .guardrails import validate_input, validate_output
    
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    # Extended state to include validation results
    class GuardrailsAgentState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]
        validation_results: Optional[List[Dict[str, Any]]]
        blocked: bool
    
    def input_validation_node(state: GuardrailsAgentState) -> Dict[str, Any]:
        """Validate user input before processing."""
        messages = state.get("messages", [])
        if not messages:
            return {"blocked": False}
        
        last_message = messages[-1]
        
        if isinstance(last_message, HumanMessage) and input_guard:
            try:
                result = validate_input(
                    input_guard,
                    last_message.content,
                    raise_on_failure=strict_mode
                )
                
                if not result["validation_passed"]:
                    logger.warning(f"Input validation failed: {result.get('error', 'Unknown error')}")
                    error_msg = AIMessage(
                        content=f"I cannot process this request. {result.get('error', 'Input validation failed.')}"
                    )
                    return {
                        "messages": [error_msg],
                        "blocked": True,
                        "validation_results": [{
                            "type": "input",
                            "passed": False,
                            "error": result.get("error")
                        }]
                    }
                
                return {
                    "blocked": False,
                    "validation_results": [{
                        "type": "input",
                        "passed": True
                    }]
                }
            except Exception as e:
                logger.error(f"Input validation error: {e}")
                if strict_mode:
                    error_msg = AIMessage(
                        content="I cannot process this request due to a validation error."
                    )
                    return {
                        "messages": [error_msg],
                        "blocked": True,
                        "validation_results": [{
                            "type": "input",
                            "passed": False,
                            "error": str(e)
                        }]
                    }
        
        return {"blocked": False}
    
    def call_model(state: GuardrailsAgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        if state.get("blocked", False):
            return {}
        
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def output_validation_node(state: GuardrailsAgentState) -> Dict[str, Any]:
        """Validate agent output before returning."""
        messages = state.get("messages", [])
        if not messages:
            return {}
        
        last_message = messages[-1]
        
        if isinstance(last_message, AIMessage) and output_guard:
            try:
                result = validate_output(
                    output_guard,
                    last_message.content,
                    raise_on_failure=strict_mode
                )
                
                if not result["validation_passed"]:
                    logger.warning(f"Output validation failed: {result.get('error', 'Unknown error')}")
                    # Replace with safe response
                    safe_response = AIMessage(
                        content="I apologize, but I cannot provide that response as it doesn't meet our safety guidelines."
                    )
                    return {
                        "messages": [safe_response],
                        "validation_results": [{
                            "type": "output",
                            "passed": False,
                            "error": result.get("error")
                        }]
                    }
                
                return {
                    "validation_results": [{
                        "type": "output",
                        "passed": True
                    }]
                }
            except Exception as e:
                logger.error(f"Output validation error: {e}")
                if strict_mode:
                    safe_response = AIMessage(
                        content="I apologize, but I cannot provide that response due to a validation error."
                    )
                    return {
                        "messages": [safe_response],
                        "validation_results": [{
                            "type": "output",
                            "passed": False,
                            "error": str(e)
                        }]
                    }
        
        return {}
    
    def should_continue(state: GuardrailsAgentState):
        """Route to tools if the last message has tool calls."""
        if state.get("blocked", False):
            return END
        
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    def should_validate_output(state: GuardrailsAgentState):
        """Check if we should validate output."""
        if state.get("blocked", False):
            return END
        
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
            return "validate_output" if output_guard else END
        return END
    
    # Build graph
    graph = StateGraph(GuardrailsAgentState)
    tool_node = ToolNode(tools)
    
    # Add nodes
    if input_guard:
        graph.add_node("validate_input", input_validation_node)
        graph.set_entry_point("validate_input")
        graph.add_conditional_edges(
            "validate_input",
            lambda s: END if s.get("blocked", False) else "agent",
            {END: END, "agent": "agent"}
        )
    else:
        graph.set_entry_point("agent")
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    
    if output_guard:
        graph.add_node("validate_output", output_validation_node)
        # Route from agent: if tool calls, go to action; else validate output
        def route_from_agent(state: GuardrailsAgentState):
            if state.get("blocked", False):
                return END
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "action"
            return "validate_output"
        
        graph.add_conditional_edges(
            "agent",
            route_from_agent,
            {"action": "action", "validate_output": "validate_output", END: END}
        )
        graph.add_edge("validate_output", END)
    else:
        graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    
    graph.add_edge("action", "agent")
    
    return graph.compile()
