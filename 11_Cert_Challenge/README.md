# Certification Challenge — [MediQuery AI]

## 1) Problem & Audience
Problem: Patients struggle to understand dense, highly technical FDA drug labels (Patient Information Leaflets), leading to medication errors, poor adherence, and significant safety risks.

Audience: This tool is for patients, their families, and caregivers. This audience, particularly the elderly or those with lower health literacy, is often given a multi-page document filled with medical jargon. They have simple, critical questions ("Can I take this with alcohol?", "What's the most common side effect?", "What happens if I miss a dose?"), but finding the answer is like finding a needle in a haystack.

MediQuery bridges this gap by acting as a "pharmacist-in-your-pocket." It ingests the official, complex data and allows a user to ask simple questions in plain English. It provides clear, sourced answers, helping users safely manage their medications and improving health outcomes.

## 2) Proposed Solution & Stack
### LLM: OpenAI gpt-4o
### Justification: In the medical domain, accuracy and safety are non-negotiable. gpt-4o's advanced reasoning is required to correctly interpret and simplify complex medical concepts like "contraindications" vs. "adverse reactions" and to avoid dangerous hallucinations.

### Embeddings: text-embedding-3-small (OpenAI)
### Justification: This model offers the best balance of high retrieval performance on benchmarks and cost-efficiency. This is crucial for embedding a massive corpus, such as the entire database of FDA drug labels.

### Orchestrator: LangChain
### Justification: LangChain is the glue. Its DocumentLoaders are essential for parsing the highly structured (XML/JSON) data from the drug labels, and its TextSplitters are needed for the intelligent chunking strategy.

### Vector DB: QDrant
### Justification: The full DailyMed (all FDA labels) dataset is enormous, containing millions of document sections. Currently using local version.

### Eval: RAGAS
### Justification: RAGAS is purpose-built for this task. The Faithfulness metric is the single most important KPI, as it measures whether the LLM's simplified answer is factually grounded in the retrieved drug label sections, preventing the system from "inventing" side effects or dosages.

### Validators: Guardrails AI
### Justification: This is a critical safety component. Guardrails AI will be configured to immediately intercept any query that constitutes asking for medical advice (e.g., "Should I stop taking my pill?," "Is 20mg too much for me?"). These queries must be blocked from reaching the LLM and instead return a hard-coded response: "I cannot provide medical advice. Please consult your doctor."

### UI: Streamlit
### Justification: Streamlit is the fastest way to build this prototype. Its st.selectbox widget is perfect for drug selection, and st.chat_input provides the Q&A interface instantly.


## 3) Data & Chunking
### Data Sources/APIs: 
### Primary Source: DailyMed (via NLM/FDA)- This is a public database containing the official, structured (XML/SPL) "content of labeling" for all FDA-approved drugs. This is our core knowledge base.


## 4) End-to-End Prototype

### Architecture Flow
User Query → LangGraph State → Retrieve Node → Generate Node → Response
     ↓              ↓              ↓              ↓
  Streamlit    StateGraph    Vector Search    LLM Generation

### How to run the app?
1. Install dependencies i.e. uv sync
2. Update secrets in the .streamlit/secrets.toml
3. Open terminal and run app i.e. streamlit run app.py

## 5) Golden Test Set & RAGAS
### 📊 **RAGAS Evaluation Results Summary**

### **Performance Comparison: Baseline vs. Reranked Retrieval**

| Metric | Baseline | Reranked | Improvement | Impact |
|--------|----------|----------|-------------|---------|
| **Context Recall** | 62.00% | 80.33% | **+18.33%** | 🟢 **High** |
| **Faithfulness** | 67.42% | 77.04% | **+9.62%** | 🟡 **Medium** |
| **Factual Correctness** | 50.00% | 68.50% | **+18.50%** | 🟢 **High** |
| **Answer Relevancy** | 85.50% | 94.91% | **+9.40%** | 🟡 **Medium** |
| **Context Entity Recall** | 45.60% | 68.61% | **+23.00%** | 🟢 **Highest** |
| **Noise Sensitivity** | 9.33% | 12.52% | **+3.19%** | 🔴 **Low** |

### **Key Insights**

#### **🎯 Top Improvements**
1. **Context Entity Recall**: +23.00% - Reranking significantly better at finding relevant entities
2. **Context Recall**: +18.33% - Much better at retrieving relevant context chunks
3. **Factual Correctness**: +18.50% - Substantial improvement in factual accuracy

#### **📈 Performance Highlights**
- **Faithfulness**: Improved from 67.4% to 77.0% - Critical for medical safety
- **Answer Relevancy**: Already high at 85.5%, improved to 94.9% - Excellent user experience
- **Overall**: Reranking shows consistent improvements across all metrics

#### **🔍 Analysis**
- **Strongest Impact**: Entity and context recall improvements suggest reranking better identifies relevant medical entities and context
- **Safety Critical**: Faithfulness improvement is crucial for medical applications
- **User Experience**: High answer relevancy ensures responses match user intent
- **Minimal Noise**: Low noise sensitivity increase (+3.19%) indicates reranking doesn't introduce significant irrelevant information

### **Conclusion**
The reranked retrieval approach demonstrates **significant improvements** across all RAGAS metrics, with particularly strong gains in entity recall and factual correctness - both critical for medical applications where accuracy and completeness are paramount.

## 6) Advanced Retrieval
TBD

## 7) Performance & Next Steps
Persistent Vector Store: The current vector store (:memory:) re-builds every time  restart the app. Change QdrantClient(":memory:") to QdrantClient(path="./qdrant_db") to save the index to disk. This will make restarts much faster.

Integrate Real-Time API: Supplement the static DailyMed knowledge base by adding a tool that can call the openFDA API in real-time. This would allow the bot to answer questions like "Are there any recent recalls for my drug?"—information that is not in the static dataset
---

> 🧭 Deliverables: see [`/deliverables/`](deliverables/) for the generated checklist and slide outline.
