# Domain-Specific RAG Chatbot for PDF Question Answering

## Features
- Multiple PDF upload; page-aware text extraction with pypdf
- Character chunking with overlap and source/page metadata
- `all-MiniLM-L6-v2` sentence embeddings
- FAISS inner-product search on normalized vectors (cosine similarity)
- Top-4 retrieval, source page display and no-answer fallback
- Streamlit chat UI

## Run locally
1. Install Python 3.10+ and open a terminal in this folder.
2. Create/activate a virtual environment (commands in bundle README).
3. Install: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Upload a text-based PDF, click Process Documents, then ask a question.

## Important implementation note
This starter is a grounded extractive RAG MVP: it retrieves relevant passages and presents them as evidence rather than calling an external generative LLM. The project guidance lists LLM generation as a pipeline stage; add a secured API/local model later if required. This approach is runnable without an API key and helps avoid unsupported answers.

## Limitations / safety
Scanned PDFs need OCR (not included). Similarity scores are retrieval relevance estimates, not truth scores. Verify important answers in the original document. Never upload confidential documents without permission.
