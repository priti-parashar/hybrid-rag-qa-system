import streamlit as st
import pypdf
import re
import numpy as np

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG PDF Q&A",
    page_icon="📄"
)

st.title("📄 Ask Your PDF")
st.caption("Hybrid RAG: Semantic Search + BM25 + CrossEncoder Reranking")

CONFIDENCE_THRESHOLD = 0.1


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    # Embedding model for semantic search
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # CrossEncoder for reranking retrieved chunks
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Extractive question-answering model
    qa = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )

    return model, reranker, qa


# --------------------------------------------------
# DEFINITION-BASED CHUNKING
# --------------------------------------------------

def chunk_by_definition(text):

    pattern = re.compile(
        r'([A-Za-z0-9_@\.\*\(\)\s/]{2,60})\n'
        r'Definition:\s*(.*?)\n'
        r'Example:\n'
        r'(.*?)'
        r'(?=\n[A-Za-z0-9_@\.\*\(\)\s/]{2,60}\nDefinition:|\Z)',
        re.DOTALL
    )

    chunks = []

    for term, definition, example in pattern.findall(text):

        chunk = (
            f"{term.strip()}\n"
            f"Definition: {definition.strip()}\n"
            f"Example:\n{example.strip()}"
        )

        chunks.append(chunk)

    return chunks


# --------------------------------------------------
# GENERAL SENTENCE CHUNKING
# --------------------------------------------------

def chunk_by_sentences(
    text,
    sentences_per_chunk=3,
    overlap=1
):

    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    i = 0

    step = max(
        1,
        sentences_per_chunk - overlap
    )

    while i < len(sentences):

        chunk_sentences = sentences[
            i:i + sentences_per_chunk
        ]

        if chunk_sentences:
            chunks.append(
                " ".join(chunk_sentences)
            )

        i += step

    return chunks


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

@st.cache_data
def process_pdf(file_bytes):

    pdf = pypdf.PdfReader(file_bytes)

    text = ""

    for page in pdf.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # First try definition-based chunking
    chunks = chunk_by_definition(text)

    used_fallback = False

    # If the document does not follow the
    # Definition / Example structure,
    # use general sentence chunking.
    if not chunks:

        chunks = chunk_by_sentences(text)

        used_fallback = True

    if not chunks:
        return [], np.array([]), None, used_fallback

    # Generate semantic embeddings
    model, _, _ = load_models()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    # Prepare chunks for BM25
    tokenized_chunks = [
        re.findall(
            r'\w+\b',
            chunk.lower()
        )
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return (
        chunks,
        embeddings,
        bm25,
        used_fallback
    )


# --------------------------------------------------
# HYBRID RETRIEVAL + RERANKING + QA
# --------------------------------------------------

def answer_question(
    query,
    chunks,
    embeddings,
    bm25,
    model,
    reranker,
    qa
):

    # ---------------------------
    # 1. SEMANTIC SEARCH
    # ---------------------------

    query_embedding = model.encode(query)

    scores = util.cos_sim(
        query_embedding,
        embeddings
    ).squeeze(0)

    semantic_scores = (
        scores.cpu()
        .numpy()
        .flatten()
    )

    # ---------------------------
    # 2. BM25 SEARCH
    # ---------------------------

    query_tokens = re.findall(
        r'\w+\b',
        query.lower()
    )

    bm25_scores = np.array(
        bm25.get_scores(query_tokens)
    )

    # ---------------------------
    # 3. NORMALIZE SCORES
    # ---------------------------

    semantic_range = (
        semantic_scores.max()
        - semantic_scores.min()
    )

    if semantic_range == 0:

        semantic_normalized = np.zeros_like(
            semantic_scores
        )

    else:

        semantic_normalized = (
            semantic_scores
            - semantic_scores.min()
        ) / semantic_range

    bm25_range = (
        bm25_scores.max()
        - bm25_scores.min()
    )

    if bm25_range == 0:

        bm25_normalized = np.zeros_like(
            bm25_scores
        )

    else:

        bm25_normalized = (
            bm25_scores
            - bm25_scores.min()
        ) / bm25_range

    # ---------------------------
    # 4. HYBRID SCORE
    # ---------------------------

    hybrid_scores = (
        0.5 * semantic_normalized
        + 0.5 * bm25_normalized
    )

    # Retrieve top 5 candidate chunks
    top_indices = (
        hybrid_scores
        .argsort()[-5:][::-1]
    )

    # ---------------------------
    # 5. CROSSENCODER RERANKING
    # ---------------------------

    pairs = [
        (
            query,
            chunks[int(index)]
        )
        for index in top_indices
    ]

    rerank_scores = reranker.predict(pairs)

    ranked_positions = np.argsort(
        rerank_scores
    )[::-1]

    # ---------------------------
    # 6. USE TOP 3 CHUNKS
    # ---------------------------

    best_indices = [
        int(top_indices[position])
        for position in ranked_positions[:3]
    ]

    selected_chunks = [
        chunks[index]
        for index in best_indices
    ]

    # Combine the best chunks
    context = "\n\n".join(
        selected_chunks
    )

    # ---------------------------
    # 7. QUESTION ANSWERING
    # ---------------------------

    result = qa(
        question=query,
        context=context,
        max_answer_len=100
    )

    return result, context


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)


if uploaded_file:

    try:

        with st.spinner(
            "Processing PDF..."
        ):

            chunks, embeddings, bm25, used_fallback = (
                process_pdf(uploaded_file)
            )

            model, reranker, qa = load_models()

        if not chunks:

            st.error(
                "Couldn't extract readable text from this PDF. "
                "It may be scanned or image-only."
            )

            st.stop()

    except Exception as e:

        st.error(
            f"Something went wrong processing this PDF: {e}"
        )

        st.stop()


    # PDF successfully processed
    st.success(
        f"Ready! Extracted {len(chunks)} chunks."
    )


    if used_fallback:

        st.caption(
            "Using general-purpose sentence-based "
            "chunking for this document."
        )


    # --------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------

    query = st.text_input(
        "Ask a question about the PDF:"
    )


    if query:

        try:

            with st.spinner(
                "Searching the document..."
            ):

                result, context = answer_question(
                    query,
                    chunks,
                    embeddings,
                    bm25,
                    model,
                    reranker,
                    qa
                )

            # ------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------

            st.subheader("Answer")


            if result["score"] < CONFIDENCE_THRESHOLD:

                st.warning(
                    "I'm not confident enough to answer "
                    "this from the document. Try rephrasing "
                    "your question."
                )

            else:

                st.write(
                    result["answer"]
                )


            # Confidence score
            st.caption(
                f"Confidence: {result['score']:.3f}"
            )


            # ------------------------------------------
            # SOURCE CONTEXT
            # ------------------------------------------

            with st.expander(
                "Show source context"
            ):

                st.text(context)


        except Exception as e:

            st.error(
                f"Something went wrong while answering: {e}"
            )
