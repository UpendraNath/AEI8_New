# --- 1. Imports ---
# Standard library imports
import os
import zipfile
from pathlib import Path
from getpass import getpass
from typing_extensions import List, TypedDict

# Streamlit - for the UI
import streamlit as st

# Third-party imports (from your notebook)
import pandas as pd
from bs4 import BeautifulSoup
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# --- 2. Setup API Keys ---
# Use Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
# Optional: LangSmith for tracing
os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", "")
if os.environ["LANGSMITH_API_KEY"]:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "AIM - RAG App"

# --- 3. Re-usable Data Loading Functions (Copied from your notebook) ---

def unzip_data_files(data_dir: Path) -> None:
    """Unzip all zip files found in data directory"""
    zip_files = list(data_dir.glob("*.zip"))
    st.write(f"Found {len(zip_files)} zip files to extract...")
    for zip_file in zip_files:
        extract_dir = zip_file.parent / zip_file.stem
        if not extract_dir.exists():  # Only extract if not already done
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                st.write(f"✓ Extracted {zip_file.name}")
            except Exception as e:
                st.error(f"✗ Failed to extract {zip_file.name}: {str(e)}")
        else:
            st.write(f"✓ Directory already exists: {extract_dir.name}")

def find_xml_files(data_dir: Path) -> List[Path]:
    """Find all XML files in data directory and its subdirectories"""
    return list(data_dir.rglob("*.xml"))

def extract_drug_info_from_xml(xml_file: Path, data_path: Path) -> dict:
    """Extract drug name and other metadata from XML file"""
    try:
        with open(xml_file) as f:
            soup = BeautifulSoup(f, 'xml')
        
        drug_name = "Unknown Drug"
        title_tag = soup.find('title')
        if title_tag and title_tag.text:
            drug_name = title_tag.text.strip()
        else:
            name_tag = soup.find('name')
            if name_tag and name_tag.text:
                drug_name = name_tag.text.strip()
        
        section = "General"
        section_tag = soup.find('section')
        if section_tag and section_tag.text:
            section = section_tag.text.strip()
        
        return {
            'drug_name': drug_name,
            'section': section,
            'file_path': str(xml_file.relative_to(data_path))
        }
    except Exception:
        return {
            'drug_name': 'Unknown Drug',
            'section': 'General',
            'file_path': str(xml_file.relative_to(data_path))
        }

def extract_text_from_xml(xml_file: Path, data_path: Path) -> List[dict]:
    """Extract text content from XML file"""
    with open(xml_file) as f:
        soup = BeautifulSoup(f, 'xml')
    
    metadata = extract_drug_info_from_xml(xml_file, data_path)
    paragraphs = []
    current_text = ""
    min_length = 200  # Minimum characters for a document
    
    for p in soup.find_all('paragraph'):
        if p.text.strip():
            text = p.text.strip()
            if len(text) > 50:
                current_text += text + " "
                if len(current_text) > min_length:
                    paragraphs.append({
                        'text': current_text.strip(),
                        'source': metadata['file_path'],
                        'drug_name': metadata['drug_name'],
                        'section': metadata['section']
                    })
                    current_text = ""
    
    if len(current_text.strip()) > min_length:
        paragraphs.append({
            'text': current_text.strip(),
            'source': metadata['file_path'],
            'drug_name': metadata['drug_name'],
            'section': metadata['section']
        })
    return paragraphs

# --- 4. The Main RAG Chain (Cached) ---
# @st.cache_resource is the *most important* part.
# It tells Streamlit to run this complex function ONCE and store the result
# (our RAG chain) in memory. This prevents re-loading and re-indexing
# all the data on every single user message.

@st.cache_resource
def get_rag_chain():
    """
    This function handles all the one-time setup:
    1. Load and process data
    2. Split and embed documents
    3. Initialize the Vector DB (Qdrant)
    4. Compile the LangGraph (the RAG chain)
    """
    st.write("Setting up RAG chain... (This runs only once)")
    
    # --- A. Load and Process Data ---
    # Find the data directory (assuming app.py is in the same dir as the notebook)
    project_root = Path.cwd().parent
    data_path = project_root / "data/raw"
    st.write(f"Data path: {data_path}")

    unzip_data_files(data_path)
    xml_files = find_xml_files(data_path)
    
    all_paragraphs = []
    for xml_file in xml_files:
        all_paragraphs.extend(extract_text_from_xml(xml_file, data_path))
        
    df = pd.DataFrame(all_paragraphs)
    df = df[df['text'].str.len() > 100] # Filter short docs
    st.write(f"Loaded {len(df)} documents.")

    # Convert to LangChain Document objects
    documents = [
        Document(
            page_content=row['text'],
            metadata={
                'source': row['source'],
                'drug_name': row['drug_name'], 
                'section': row['section']
            }
        ) for _, row in df.iterrows()
    ]

    # --- B. Chunk and Embed ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    split_documents = text_splitter.split_documents(documents)
    st.write(f"Created {len(split_documents)} document chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- C. Setup Vector Store (Qdrant) ---
    # We use an in-memory Qdrant store
    baseline_client = QdrantClient(":memory:")
    baseline_client.create_collection(
        collection_name="drug_labels",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    baseline_vector_store = QdrantVectorStore(
        client=baseline_client,
        collection_name="drug_labels",
        embedding=embeddings,
    )
    baseline_vector_store.add_documents(documents=split_documents)
    retriever = baseline_vector_store.as_retriever(search_kwargs={"k": 3})
    st.write("✅ Vector store is ready.")

    # --- D. Define the LangGraph State ---
    class State(TypedDict):
        question: str
        context: List[Document]
        response: str

    # --- E. Define Graph Nodes ---
    llm = ChatOpenAI(model="gpt-4.1-nano")
    BASELINE_PROMPT = """
    You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

    ### Question
    {question}

    ### Context
    {context}
    """
    rag_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

    def retrieve(state):
        retrieved_docs = retriever.invoke(state["question"])
        return {"context" : retrieved_docs}

    def generate(state):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
        response = llm.invoke(messages)
        return {"response" : response.content}

    # --- F. Compile the Graph ---
    baseline_graph_builder = StateGraph(State)
    baseline_graph_builder.add_node("retrieve", retrieve)
    baseline_graph_builder.add_node("generate", generate)
    baseline_graph_builder.add_edge(START, "retrieve")
    baseline_graph_builder.add_edge("retrieve", "generate")
    baseline_graph = baseline_graph_builder.compile()
    
    st.write("✅ RAG chain compiled.")
    return baseline_graph


# --- 5. The Streamlit UI ---

st.title("💊 MediQuery AI Chatbot")
st.markdown("Ask me questions about the drug information in our database.")

# Initialize the RAG chain
try:
    rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Failed to initialize RAG chain: {e}")
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is OCREVUS ZUNOVO used for?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        response_container = st.empty() # Placeholder for streaming
        full_response = ""
        
        # Use .stream() to get live events from the LangGraph
        events = rag_chain.stream(
            {"question": prompt},
            # Stream all node outputs
            stream_mode="values"
        )
        
        for event in events:
            # The event is the full state dict. We look for the "response" key.
            if "response" in event:
                full_response = event["response"]
                response_container.markdown(full_response + "▌") # Add cursor
        
        # Final update to remove cursor
        response_container.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})