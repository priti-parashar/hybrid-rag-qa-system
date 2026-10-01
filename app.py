import streamlit as st
import pypdf
import re
import numpy as np

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG PDF Q&A",
    page_icon="📄"
)

st.title("📄 Hybrid RAG PDF Q&A")
st.caption("Semantic Search + BM25 + CrossEncoder Reranking")


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return embedding_model, reranker


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

def chunk_text(
    text,
    words_per_chunk=180,
    overlap=40
):

    words = text.split()

    if not words:
        return []

    chunks = []

    step = words_per_chunk - overlap

    for start in range(
        0,
        len(words),
        step
    ):

        chunk_words = words[
            start:start + words_per_chunk
        ]

        if not chunk_words:
            continue

        chunk = " ".join(
            chunk_words
        ).strip()

        if chunk:
            chunks.append(chunk)

        if (
            start + words_per_chunk
            >= len(words)
        ):
            break

    return chunks


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

@st.cache_data
def process_pdf(file_bytes):

    pdf = pypdf.PdfReader(
        file_bytes
    )

    pages = []

    for page in pdf.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    text = "\n\n".join(pages)

    text = clean_text(text)

    chunks = chunk_text(text)

    if not chunks:
        return [], np.array([]), None

    embedding_model, _ = load_models()

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    tokenized_chunks = [
        re.findall(
            r"\b\w+\b",
            chunk.lower()
        )
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    return chunks, embeddings, bm25


# --------------------------------------------------
# NORMALIZE SCORES
# --------------------------------------------------

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    score_min = scores.min()
    score_max = scores.max()

    if score_max == score_min:
        return np.zeros_like(
            scores,
            dtype=float
        )

    return (
        scores - score_min
    ) / (
        score_max - score_min
    )


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

def retrieve_chunks(
    query,
    chunks,
    embeddings,
    bm25,
    embedding_model,
    reranker,
    top_k=5
):

    # Semantic search
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    semantic_scores = util.cos_sim(
        query_embedding,
        embeddings
    )[0].cpu().numpy()

    # BM25 search
    query_tokens = re.findall(
        r"\b\w+\b",
        query.lower()
    )

    bm25_scores = np.asarray(
        bm25.get_scores(
            query_tokens
        )
    )

    # Normalize scores
    semantic_normalized = normalize_scores(
        semantic_scores
    )

    bm25_normalized = normalize_scores(
        bm25_scores
    )

    # Hybrid score
    hybrid_scores = (
        0.6 * semantic_normalized
        + 0.4 * bm25_normalized
    )

    candidate_count = min(
        10,
        len(chunks)
    )

    candidate_indices = np.argsort(
        hybrid_scores
    )[-candidate_count:][::-1]

    # CrossEncoder reranking
    pairs = [
        (
            query,
            chunks[int(index)]
        )
        for index in candidate_indices
    ]

    rerank_scores = np.asarray(
        reranker.predict(pairs)
    )

    reranked_positions = np.argsort(
        rerank_scores
    )[::-1]

    final_indices = [
        int(
            candidate_indices[position]
        )
        for position
        in reranked_positions[:top_k]
    ]

    final_chunks = [
        chunks[index]
        for index in final_indices
    ]

    return final_chunks


# --------------------------------------------------
# EXTRACT SEARCH SUBJECT
# --------------------------------------------------

def extract_subject(query):

    query_lower = query.lower().strip()

    patterns = [
        r"time complexity of (.+?)(?:\?|$)",
        r"space complexity of (.+?)(?:\?|$)",
        r"complexity of (.+?)(?:\?|$)",
        r"what is (.+?)(?:\?|$)",
        r"what are (.+?)(?:\?|$)",
        r"define (.+?)(?:\?|$)",
        r"explain (.+?)(?:\?|$)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query_lower
        )

        if match:

            subject = match.group(1).strip()

            subject = re.sub(
                r"\bthe\b",
                "",
                subject
            ).strip()

            if subject:
                return subject

    return query_lower.rstrip("?")


# --------------------------------------------------
# COMPLEXITY ANSWER
# --------------------------------------------------

def find_complexity_answer(
    query,
    retrieved_chunks
):

    query_lower = query.lower()

    if not (
        "complexity" in query_lower
        or "big o" in query_lower
        or "big-o" in query_lower
    ):
        return None

    subject = extract_subject(query)

    # Example:
    # subject = "binary search"
    subject_pattern = re.escape(subject)

    # Look for:
    # Binary search ... O(log n)
    forward_pattern = re.compile(
        rf"{subject_pattern}"
        rf".{{0,100}}?"
        rf"(O\s*\(\s*[^)]+\s*\))",
        re.IGNORECASE
    )

    # Also handle:
    # O(log n) ... binary search
    backward_pattern = re.compile(
        rf"(O\s*\(\s*[^)]+\s*\))"
        rf".{{0,100}}?"
        rf"{subject_pattern}",
        re.IGNORECASE
    )

    for chunk in retrieved_chunks:

        forward_match = forward_pattern.search(
            chunk
        )

        if forward_match:

            complexity = forward_match.group(1)

            complexity = re.sub(
                r"\s+",
                " ",
                complexity
            )

            return (
                f"The time complexity of "
                f"{subject} is {complexity}."
            )

        backward_match = backward_pattern.search(
            chunk
        )

        if backward_match:

            complexity = backward_match.group(1)

            complexity = re.sub(
                r"\s+",
                " ",
                complexity
            )

            return (
                f"The time complexity of "
                f"{subject} is {complexity}."
            )

    return None


# --------------------------------------------------
# SPLIT CONTEXT INTO SMALL PASSAGES
# --------------------------------------------------

def make_passages(text):

    # PDF extraction often removes punctuation/newlines.
    # Therefore we create smaller overlapping word passages
    # instead of relying only on sentence boundaries.

    words = text.split()

    passages = []

    passage_size = 35
    overlap = 10

    step = passage_size - overlap

    for start in range(
        0,
        len(words),
        step
    ):

        passage_words = words[
            start:start + passage_size
        ]

        if not passage_words:
            continue

        passage = " ".join(
            passage_words
        ).strip()

        if passage:
            passages.append(passage)

        if (
            start + passage_size
            >= len(words)
        ):
            break

    return passages


# --------------------------------------------------
# GENERAL ANSWER
# --------------------------------------------------

def find_general_answer(
    query,
    retrieved_chunks,
    embedding_model
):

    passages = []

    for chunk in retrieved_chunks:

        passages.extend(
            make_passages(chunk)
        )

    if not passages:
        return None, 0.0

    # Remove duplicate passages
    unique_passages = []
    seen = set()

    for passage in passages:

        key = passage.lower()

        if key not in seen:

            seen.add(key)

            unique_passages.append(
                passage
            )

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    passage_embeddings = embedding_model.encode(
        unique_passages,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(
        query_embedding,
        passage_embeddings
    )[0].cpu().numpy()

    best_index = int(
        np.argmax(scores)
    )

    best_score = float(
        scores[best_index]
    )

    best_passage = unique_passages[
        best_index
    ]

    return best_passage, best_score


# --------------------------------------------------
# CREATE FINAL ANSWER
# --------------------------------------------------

def create_answer(
    query,
    retrieved_chunks,
    embedding_model
):

    # First handle complexity questions
    complexity_answer = find_complexity_answer(
        query,
        retrieved_chunks
    )

    if complexity_answer:

        return (
            complexity_answer,
            1.0
        )

    # Otherwise use semantic passage selection
    answer, score = find_general_answer(
        query,
        retrieved_chunks,
        embedding_model
    )

    return answer, score


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    try:

        with st.spinner(
            "Processing PDF..."
        ):

            chunks, embeddings, bm25 = process_pdf(
                uploaded_file
            )

            embedding_model, reranker = load_models()

        if not chunks:

            st.error(
                "No readable text was found in this PDF. "
                "The PDF may be scanned or image-only."
            )

            st.stop()

        st.success(
            f"Ready! Extracted {len(chunks)} chunks."
        )

    except Exception as error:

        st.error(
            f"Something went wrong while processing "
            f"the PDF: {error}"
        )

        st.stop()


    query = st.text_input(
        "Ask a question about the PDF:"
    )


    if query:

        try:

            with st.spinner(
                "Searching the document..."
            ):

                retrieved_chunks = retrieve_chunks(
                    query,
                    chunks,
                    embeddings,
                    bm25,
                    embedding_model,
                    reranker
                )

                answer, confidence = create_answer(
                    query,
                    retrieved_chunks,
                    embedding_model
                )


            st.subheader("Answer")


            if (
                answer is None
                or confidence < 0.20
            ):

                st.warning(
                    "I couldn't find a reliable answer "
                    "to this question in the document."
                )

            else:

                st.write(answer)


            st.caption(
                f"Answer relevance: {confidence:.3f}"
            )


            # --------------------------------------------------
            # SOURCE CONTEXT
            # --------------------------------------------------

            with st.expander(
                "Show source context"
            ):

                for number, chunk in enumerate(
                    retrieved_chunks[:3],
                    start=1
                ):

                    st.markdown(
                        f"**Source {number}**"
                    )

                    st.write(chunk)

                    if number < min(
                        3,
                        len(retrieved_chunks)
                    ):

                        st.divider()


        except Exception as error:

            st.error(
                f"Something went wrong while answering: "
                f"{error}"
            )
