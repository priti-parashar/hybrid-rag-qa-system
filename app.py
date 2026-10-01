import streamlit as st
import pypdf
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 Ask Your PDF")

CONFIDENCE_THRESHOLD = 0.1

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    qa = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )
    return model, reranker, qa

def chunk_by_definition(text):
    pattern = re.compile(
        r'([A-Za-z0-9_@\.\*\(\)\s/]{2,60})\nDefinition:\s*(.*?)\nExample:\n(.*?)(?=\n[A-Za-z0-9_@\.\*\(\)\s/]{2,60}\nDefinition:|\Z)',
        re.DOTALL
    )
    chunks = []
    for term, definition, example in pattern.findall(text):
        chunk = f"{term.strip()}\nDefinition: {definition.strip()}\nExample:\n{example.strip()}"
        chunks.append(chunk)
    return chunks

def chunk_by_sentences(text, sentences_per_chunk=3, overlap=1):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    i = 0
    step = max(1, sentences_per_chunk - overlap)
    while i < len(sentences):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        if chunk_sentences:
            chunks.append(" ".join(chunk_sentences))
        i += step
    return chunks

@st.cache_data
def process_pdf(file_bytes):
    pdf = pypdf.PdfReader(file_bytes)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

    chunks = chunk_by_definition(text)
    used_fallback = False
    if not chunks:
        chunks = chunk_by_sentences(text)
        used_fallback = True

    model, _, _ = load_models()
    embeddings = model.encode(chunks)

    tokenized_chunks = [re.findall(r'\w+\b', chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return chunks, embeddings, bm25, used_fallback

def answer_question(query, chunks, embeddings, bm25, model, reranker, qa):
    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, embeddings).squeeze(0)

    query_tokens = re.findall(r'\w+\b', query.lower())
    bm25_scores = np.array(bm25.get_scores(query_tokens))

    semantic_scores = scores.cpu().numpy().flatten()

    semantic_range = semantic_scores.max() - semantic_scores.min()
    semantic_normalized = (
        np.zeros_like(semantic_scores) if semantic_range == 0
        else (semantic_scores - semantic_scores.min()) / semantic_range
    )

    bm25_range = bm25_scores.max() - bm25_scores.min()
    bm25_normalized = (
        np.zeros_like(bm25_scores) if bm25_range == 0
        else (bm25_scores - bm25_scores.min()) / bm25_range
    )

    hybrid_scores = 0.5 * semantic_normalized + 0.5 * bm25_normalized
    top_indices = hybrid_scores.argsort()[-5:][::-1]

    pairs = [(query, chunks[int(i)]) for i in top_indices]
    rerank_scores = reranker.predict(pairs)

    best_position = rerank_scores.argmax()
    best_index = int(top_indices[best_position])
    context = chunks[best_index]

    result = qa(question=query, context=context)
    return result, context

# --- UI ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    try:
        with st.spinner("Processing PDF..."):
            chunks, embeddings, bm25, used_fallback = process_pdf(uploaded_file)
            model, reranker, qa = load_models()

        if not chunks:
            st.error("Couldn't extract readable text from this PDF. It may be scanned/image-only.")
            st.stop()
    except Exception as e:
        st.error(f"Something went wrong processing this PDF: {e}")
        st.stop()

    st.success(f"Ready! Extracted {len(chunks)} chunks.")
    if used_fallback:
        st.caption("Using general-purpose sentence-based chunking for this document.")

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Finding answer..."):
            result, context = answer_question(query, chunks, embeddings, bm25, model, reranker, qa)

        st.subheader("Answer")
        if result['score'] < CONFIDENCE_THRESHOLD:
            st.warning("I'm not confident enough to answer this from the document. Try rephrasing.")
        else:
            st.write(result['answer'])

        st.caption(f"Confidence: {result['score']:.3f}")

        with st.expander("Show source context"):
            st.text(context)