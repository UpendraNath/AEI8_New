"""Simple test script for the RAG application."""
import os
import sys

# Set up environment variables if not already set
if "TOGETHER_API_KEY" not in os.environ:
    print("Please set TOGETHER_API_KEY environment variable")
    sys.exit(1)

# Set default model endpoint if not set
if "TOGETHER_MODEL_ENDPOINT" not in os.environ:
    os.environ["TOGETHER_MODEL_ENDPOINT"] = "openai/gpt-oss-20b"

print("Testing RAG Application with Together AI Endpoints")
print("=" * 60)

# Test 1: Simple RAG query
print("\nTest 1: Simple RAG Query")
print("-" * 60)
try:
    from app.rag import get_rag_response
    
    query = "What is this document about?"
    model_endpoint = os.environ.get("TOGETHER_MODEL_ENDPOINT", "openai/gpt-oss-20b")
    
    print(f"Query: {query}")
    print("Processing...")
    response = get_rag_response(query, model_endpoint)
    print(f"Response: {response[:200]}...")  # Print first 200 chars
    print("✓ Test 1 passed")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Agentic RAG
print("\nTest 2: Agentic RAG with LangGraph")
print("-" * 60)
try:
    from app.agent_graph import run_agent
    
    query = "What information can you find about AI usage?"
    model_endpoint = os.environ.get("TOGETHER_MODEL_ENDPOINT", "openai/gpt-oss-20b")
    
    print(f"Query: {query}")
    print("Processing...")
    response = run_agent(query, model_endpoint)
    print(f"Response: {response[:200]}...")  # Print first 200 chars
    print("✓ Test 2 passed")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing complete!")

