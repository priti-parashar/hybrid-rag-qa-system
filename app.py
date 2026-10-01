import io
import re

import numpy as np
import pypdf
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import pipeline


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG PDF Q&A",
    page_icon="📄",
)

st.title("📄 Hybrid RAG PDF Q&A")
st.caption(
    "Semantic Search + BM25 + CrossEncoder Reranking + FLAN-T5"
)


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

    generator = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        device=-1,
    )

    return embedding_model, reranker, generator


# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# --------------------------------------------------
# CHUNK TEXT
# --------------------------------------------------

def chunk_text(
    text,
    words_per_chunk=160,
    overlap=40,
):

    words = text.split()

    if not words:
        return []

    chunks = []

    step = words_per_chunk - overlap

    for start in range(
        0,
        len(words),
        step,
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

        if start + words_per_chunk >= len(words):
            break

    return chunks


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

@st.cache_data
def process_pdf(file_bytes):

    reader = pypdf.PdfReader(
        io.BytesIO(file_bytes)
    )

    pages = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    text = "\n\n".join(pages)

    text = clean_text(text)

    chunks = chunk_text(text)

    if not chunks:
        return [], np.array([]), None

    embedding_model, _, _ = load_models()

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    tokenized_chunks = [
        re.findall(
            r"\b\w+\b",
            chunk.lower(),
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
        dtype=float,
    )

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(
            scores,
            dtype=float,
        )

    return (
        scores - minimum
    ) / (
        maximum - minimum
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
    top_k=3,
):

    # Semantic search
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    semantic_scores = util.cos_sim(
        query_embedding,
        embeddings,
    )[0].cpu().numpy()

    # BM25 keyword search
    query_tokens = re.findall(
        r"\b\w+\b",
        query.lower(),
    )

    bm25_scores = np.asarray(
        bm25.get_scores(query_tokens)
    )

    # Normalize both scores
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
        len(chunks),
    )

    candidate_indices = np.argsort(
        hybrid_scores
    )[-candidate_count:][::-1]

    # CrossEncoder reranking
    pairs = [
        (
            query,
            chunks[int(index)],
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
        int(candidate_indices[position])
        for position
        in reranked_positions[:top_k]
    ]

    final_chunks = [
        chunks[index]
        for index in final_indices
    ]

    return final_chunks


# --------------------------------------------------
# BUILD CONTEXT
# --------------------------------------------------

def build_context(
    retrieved_chunks,
    max_characters=1800,
):

    selected = []
    current_length = 0

    for chunk in retrieved_chunks:

        remaining = (
            max_characters
            - current_length
        )

        if remaining <= 0:
            break

        selected_text = chunk[:remaining]

        selected.append(
            selected_text
        )

        current_length += len(
            selected_text
        )

    return "\n\n".join(selected)


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

def generate_answer(
    query,
    retrieved_chunks,
    generator,
):

    context = build_context(
        retrieved_chunks
    )

    prompt = f"""
Answer the question using only the information
provided in the context.

Do not use outside knowledge.
Do not invent information.

If the answer cannot be found in the context,
say exactly:

I couldn't find enough information in the document.

Give a short, clear and direct answer.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False,
        truncation=True,
    )

    answer = result[0][
        "generated_text"
    ].strip()

    return answer


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_file is not None:

    try:

        file_bytes = (
            uploaded_file.getvalue()
        )

        with st.spinner(
            "Processing PDF..."
        ):

            (
                chunks,
                embeddings,
                bm25,
            ) = process_pdf(
                file_bytes
            )

            (
                embedding_model,
                reranker,
                generator,
            ) = load_models()

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
                    reranker,
                )

                answer = generate_answer(
                    query,
                    retrieved_chunks,
                    generator,
                )


            # --------------------------------------
            # ANSWER
            # --------------------------------------

            st.subheader("Answer")

            st.write(answer)


            # --------------------------------------
            # SOURCE CONTEXT
            # --------------------------------------

            with st.expander(
                "Show source context"
            ):

                for number, chunk in enumerate(
                    retrieved_chunks,
                    start=1,
                ):

                    st.markdown(
                        f"**Source {number}**"
                    )

                    st.write(chunk)

                    if number < len(
                        retrieved_chunks
                    ):
                        st.divider()


        except Exception as error:

            st.error(
                f"Something went wrong while answering: "
                f"{error}"
            )
