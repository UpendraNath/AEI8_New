"""Retrieval-Augmented Generation (RAG) utilities using Together AI endpoints.

This module builds an in-memory RAG pipeline that:
- Loads PDF documents from `RAG_DATA_DIR` (default: "data").
- Splits documents into chunks using a token-aware splitter.
- Embeds chunks with TogetherEmbeddings and stores vectors in an in-memory Qdrant store.
- Exposes a LangGraph that retrieves relevant context and generates a response.
"""
from __future__ import annotations

import os
from typing import Annotated, List

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import ChatTogether, TogetherEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


def _tiktoken_len(text: str) -> int:
    """Return token length using tiktoken; used for chunk length measurement."""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


class _RAGState(TypedDict):
    """State schema for the simple two-step RAG graph: retrieve then generate."""
    question: str
    context: List[Document]
    response: str


def _build_rag_graph(data_dir: str, model_endpoint: str):
    """Construct and compile a minimal RAG graph using Together AI endpoints.

    Steps:
    1) Load PDFs from `data_dir` recursively (best-effort).
    2) Split documents into token-aware chunks.
    3) Create embeddings with TogetherEmbeddings and an in-memory Qdrant vector store retriever.
    4) Define a chat prompt and generation model using ChatTogether.
    5) Wire a two-node graph: retrieve -> generate.
    """
    # Load PDFs from data directory (recursive)
    try:
        directory_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        documents = directory_loader.load()
    except Exception:
        documents = []

    # Split documents
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        # Fallback to legacy import path if available
        from langchain.text_splitter import (  # type: ignore
            RecursiveCharacterTextSplitter,
        )

    # Chunk size must be less than 512 tokens (embedding model limit)
    # Using 400 tokens with 50 token overlap to maintain context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, length_function=_tiktoken_len
    )
    chunks = text_splitter.split_documents(documents) if documents else []

    # Embeddings and vector store (in-memory Qdrant) using Together AI
    # Using BAAI/bge-large-en-v1.5 as recommended in ENDPOINT_SETUP.md
    embedding_model = TogetherEmbeddings(
        model="BAAI/bge-large-en-v1.5"
    )
    qdrant_vectorstore = Qdrant.from_documents(
        documents=chunks, embedding=embedding_model, location=":memory:"
    )
    # Retrieve more chunks for better context
    retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 8})

    # Prompt and model using ChatTogether
    human_template = (
        "You are a helpful assistant that answers questions based on the provided context from documents.\n\n"
        "#CONTEXT:\n{context}\n\n"
        "#QUERY:\n{query}\n\n"
        "Instructions:\n"
        "- Use the provided context to answer the query comprehensively.\n"
        "- For summary or overview questions (like 'summarize main points'), synthesize information from ALL parts of the context provided.\n"
        "- For questions about topics, examples, or main points, extract and organize the relevant information from the context.\n"
        "- The context contains real document content - use it to provide a detailed, helpful answer.\n"
        "- If you see context text above, there IS information available - use it to answer the question.\n"
        "- Only say you couldn't find information if the context section above is completely empty."
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
    
    # Use Together AI chat model
    generator_llm = ChatTogether(
        model=model_endpoint,
        temperature=0.7,
    )

    def retrieve(state: _RAGState) -> _RAGState:
        question = state["question"]
        question_lower = question.lower()
        retrieved_docs = []
        
        # Check if this is a summary/overview type query
        is_summary_query = any(keyword in question_lower for keyword in [
            "summarize", "summary", "main points", "key points", "overview", 
            "what is", "what are", "topics", "key topics"
        ])
        
        # Check if this is a general information query
        is_general_query = any(keyword in question_lower for keyword in [
            "what information", "can you find", "information about", "tell me about",
            "usage", "how is", "how are", "what can you"
        ])
        
        # For summary queries, prioritize getting diverse chunks directly
        if is_summary_query:
            # For summary queries, get a diverse sample of chunks from across the document
            # This ensures we have content to summarize even if semantic search fails
            if len(chunks) > 0:
                # Get chunks from different parts of the document
                num_chunks = len(chunks)
                # Sample every Nth chunk to get diversity
                step = max(1, num_chunks // 15)  # Get ~15 chunks spread across document
                sampled = chunks[::step][:15]
                retrieved_docs = sampled
                
                # Also try semantic search to get relevant chunks
                try:
                    semantic_docs = retriever.invoke(question) if retriever else []
                    # Add semantic results if they're different
                    seen_content = {hash(doc.page_content) for doc in retrieved_docs}
                    for doc in semantic_docs:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_content:
                            retrieved_docs.append(doc)
                            if len(retrieved_docs) >= 20:  # Limit total chunks
                                break
                except Exception:
                    pass  # If semantic search fails, use sampled chunks
        elif is_general_query:
            # For general queries, try semantic search first, then fallback
            retrieved_docs = retriever.invoke(question) if retriever else []
            
            # Also try query variations
            query_variations = []
            if "usage" in question_lower:
                if "ai" in question_lower or "chatgpt" in question_lower:
                    query_variations = ["ChatGPT usage", "AI usage patterns", "how people use", "user behavior"]
                else:
                    query_variations = [question.replace("usage", "use"), question.replace("usage", "using")]
            
            general_queries = ["document content", "key information", "main content"]
            all_queries = query_variations + general_queries
            seen_content = {hash(doc.page_content) for doc in retrieved_docs}
            
            for gen_query in all_queries:
                if len(retrieved_docs) >= 12:
                    break
                try:
                    docs = retriever.invoke(gen_query) if retriever else []
                    for doc in docs:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_content:
                            retrieved_docs.append(doc)
                            seen_content.add(content_hash)
                except Exception:
                    continue
            
            # Fallback: sample chunks if we don't have enough
            if len(retrieved_docs) < 8 and len(chunks) > 0:
                step = max(1, len(chunks) // 12)
                sampled = chunks[::step][:12]
                for doc in sampled:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        retrieved_docs.append(doc)
                        seen_content.add(content_hash)
            
            # Final fallback
            if len(retrieved_docs) == 0 and len(chunks) > 0:
                retrieved_docs = chunks[:min(15, len(chunks))]
        else:
            # For specific queries, use normal retrieval
            try:
                retrieved_docs = retriever.invoke(question) if retriever else []
            except Exception:
                retrieved_docs = []
            
            # Fallback if no results
            if len(retrieved_docs) == 0 and len(chunks) > 0:
                retrieved_docs = chunks[:8]
        
        return {"context": retrieved_docs}  # type: ignore

    def generate(state: _RAGState) -> _RAGState:
        # Format context documents as text
        context_docs = state.get("context", [])
        if context_docs:
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
        else:
            context_text = ""
        
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"query": state["question"], "context": context_text}
        )
        return {"response": response_text}  # type: ignore

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


# Cache key includes model_endpoint and data_dir to invalidate when either changes
_cache = {}

def clear_rag_cache():
    """Clear the RAG graph cache. Useful when documents or configuration change."""
    global _cache
    _cache.clear()

def _get_rag_graph(model_endpoint: str = "openai/gpt-oss-20b"):
    """Return a cached compiled RAG graph built from RAG_DATA_DIR."""
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    cache_key = (model_endpoint, data_dir)
    
    if cache_key not in _cache:
        _cache[cache_key] = _build_rag_graph(data_dir, model_endpoint)
    
    return _cache[cache_key]


def get_rag_response(query: str, model_endpoint: str = "openai/gpt-oss-20b") -> str:
    """Get a RAG response for a given query using Together AI endpoints.
    
    Args:
        query: The question to ask
        model_endpoint: The Together AI model endpoint identifier
        
    Returns:
        The generated response based on retrieved context (always a string)
    """
    graph = _get_rag_graph(model_endpoint)
    result = graph.invoke({"question": query})
    # Prefer returning the response string if available
    if isinstance(result, dict) and "response" in result:
        response = result["response"]
        return str(response) if response is not None else "I don't know."
    # Ensure we always return a string
    return str(result) if result is not None else "I don't know."
