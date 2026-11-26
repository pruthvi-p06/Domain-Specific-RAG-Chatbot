import io, os, re
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="Domain RAG Chatbot",page_icon="📚",layout="wide")
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
@st.cache_resource
def load_model(): return SentenceTransformer(MODEL_NAME)
def read_pdfs(files):
    pages=[]
    for f in files:
        reader=PdfReader(io.BytesIO(f.getvalue()))
        for pno,page in enumerate(reader.pages,1):
            text=(page.extract_text() or '').strip()
            if text: pages.append({"source":f.name,"page":pno,"text":text})
    return pages
def chunk_pages(pages,chunk_size=900,overlap=140):
    chunks=[]
    for p in pages:
        text=re.sub(r'\s+',' ',p['text']).strip()
        start=0
        while start<len(text):
            end=min(start+chunk_size,len(text)); piece=text[start:end].strip()
            if piece: chunks.append({"source":p['source'],"page":p['page'],"text":piece})
            if end==len(text): break
            start=max(0,end-overlap)
    return chunks
def build_index(chunks,model):
    vec=model.encode([c['text'] for c in chunks],normalize_embeddings=True,show_progress_bar=False)
    vec=np.asarray(vec,dtype='float32'); index=faiss.IndexFlatIP(vec.shape[1]); index.add(vec)
    return index
def retrieve(question,chunks,index,model,k=4):
    q=np.asarray(model.encode([question],normalize_embeddings=True),dtype='float32')
    scores,ids=index.search(q,min(k,len(chunks)))
    return [(float(s),chunks[int(i)]) for s,i in zip(scores[0],ids[0]) if i>=0]
def grounded_answer(question,hits):
    # Safe extractive baseline: present retrieved evidence only, without fabricating a fluent answer.
    if not hits or hits[0][0] < 0.22:
        return "I could not find this information in the uploaded documents."
    return "Here are the most relevant passages from your documents:\n\n"+'\n\n'.join(f"• {h['text']}" for score,h in hits[:3])

st.title("📚 Domain-Specific RAG Chatbot")
st.caption("Upload PDFs and ask questions. Answers are grounded in retrieved text and show document/page sources.")
with st.sidebar:
    files=st.file_uploader("Upload PDF documents",type=['pdf'],accept_multiple_files=True)
    process=st.button("Process Documents",type="primary")
    st.caption("Use non-confidential, text-based PDFs. Scanned PDFs need OCR; this MVP does not include OCR.")
if process:
    if not files: st.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("Extracting, chunking and embedding documents (first run downloads the embedding model)..."):
            pages=read_pdfs(files); chunks=chunk_pages(pages)
            if not chunks: st.error("No selectable text was found in the PDFs.")
            else:
                model=load_model(); index=build_index(chunks,model)
                st.session_state['rag_chunks']=chunks; st.session_state['rag_index']=index
                st.session_state['rag_model']=model; st.session_state['rag_names']=[f.name for f in files]
                st.session_state['rag_chat']=[]
                st.success(f"Indexed {len(chunks)} text chunks from {len(files)} PDF(s).")
if 'rag_index' in st.session_state:
    st.subheader("Ask a question")
    if st.button("Clear chat"): st.session_state['rag_chat']=[]
    for msg in st.session_state.get('rag_chat',[]):
        with st.chat_message(msg['role']): st.markdown(msg['content'])
    question=st.chat_input("Ask something from your uploaded documents...")
    if question:
        st.session_state['rag_chat'].append({'role':'user','content':question})
        hits=retrieve(question,st.session_state['rag_chunks'],st.session_state['rag_index'],st.session_state['rag_model'])
        answer=grounded_answer(question,hits)
        st.session_state['rag_chat'].append({'role':'assistant','content':answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
            if hits and hits[0][0]>=0.22:
                st.markdown("**Sources**")
                for score,chunk in hits:
                    st.caption(f"{chunk['source']} — page {chunk['page']} (similarity {score:.3f})")
else:
    st.info("Upload PDF files in the sidebar and click **Process Documents** to begin.")
st.caption("This extractive MVP does not use an external LLM. It shows retrieved evidence, reducing unsupported generation. Verify important answers against the original PDF.")
