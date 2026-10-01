import io
import re
import time

import numpy as np
import pypdf
import streamlit as st
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG PDF Q&A",
    page_icon="📄",
)

st.title("📄 Hybrid RAG PDF Q&A")
st.caption(
    "Semantic Search + BM25 + CrossEncoder Reranking + Gemini"
)


# --------------------------------------------------
# LOAD RETRIEVAL MODELS
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
# GEMINI CLIENT
# --------------------------------------------------

@st.cache_resource
def get_gemini_client():

    if "GEMINI_API_KEY" not in st.secrets:
        st.error(
            "Gemini API key is missing from Streamlit Secrets."
        )
        st.stop()

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


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

    embedding_model, _ = load_models()

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
    top_k=4,
):

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    semantic_scores = util.cos_sim(
        query_embedding,
        embeddings,
    )[0].cpu().numpy()

    query_tokens = re.findall(
        r"\b\w+\b",
        query.lower(),
    )

    bm25_scores = np.asarray(
        bm25.get_scores(
            query_tokens
        )
    )

    semantic_normalized = normalize_scores(
        semantic_scores
    )

    bm25_normalized = normalize_scores(
        bm25_scores
    )

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

    return [
        chunks[index]
        for index in final_indices
    ]


# --------------------------------------------------
# BUILD CONTEXT
# --------------------------------------------------

def build_context(
    retrieved_chunks,
    max_characters=6000,
):

    selected_chunks = []
    total_length = 0

    for chunk in retrieved_chunks:

        remaining = (
            max_characters
            - total_length
        )

        if remaining <= 0:
            break

        selected_chunk = chunk[:remaining]

        selected_chunks.append(
            selected_chunk
        )

        total_length += len(
            selected_chunk
        )

    return "\n\n---\n\n".join(
        selected_chunks
    )


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

def generate_answer(
    question,
    retrieved_chunks,
):

    client = get_gemini_client()

    context = build_context(
        retrieved_chunks
    )

    prompt = f"""
You are answering a question about an uploaded document.

Use ONLY the document context provided below.

Rules:
- Answer the user's exact question.
- Do not use outside knowledge.
- Do not invent facts.
- Combine relevant information from the context when necessary.
- Give a clear, natural and concise answer.
- Do not simply copy an unrelated sentence.
- If the document context does not contain enough information,
  reply exactly:
  "I couldn't find enough information in the document."

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ]

    last_error = None

    for model_name in models:

        for attempt in range(2):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                if response.text:
                    return response.text.strip()

            except Exception as error:

                last_error = error

                error_text = str(error).lower()

                temporary_error = any(
                    item in error_text
                    for item in [
                        "503",
                        "unavailable",
                        "high demand",
                        "429",
                        "resource_exhausted",
                        "timeout",
                    ]
                )

                if not temporary_error:
                    raise

                if attempt == 0:
                    time.sleep(2)

    raise RuntimeError(
        "Gemini is temporarily unavailable."
    ) from last_error


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    # If a different PDF is uploaded,
    # start a fresh conversation.
    pdf_identifier = (
        uploaded_file.name,
        len(file_bytes),
    )

    if (
        st.session_state.current_pdf
        != pdf_identifier
    ):

        st.session_state.current_pdf = (
            pdf_identifier
        )

        st.session_state.messages = []

    try:

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
            ) = load_models()

        if not chunks:

            st.error(
                "No readable text was found in this PDF. "
                "It may be scanned or image-only."
            )

            st.stop()

        st.success(
            f"Ready! Extracted {len(chunks)} chunks."
        )

    except Exception as error:

        st.error(
            "Something went wrong while processing "
            f"the PDF: {error}"
        )

        st.stop()


    # --------------------------------------------------
    # DISPLAY PREVIOUS CHAT
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "Show source context"
                ):

                    for number, chunk in enumerate(
                        message["sources"],
                        start=1,
                    ):

                        st.markdown(
                            f"**Source {number}**"
                        )

                        st.write(chunk)

                        if number < len(
                            message["sources"]
                        ):
                            st.divider()


    # --------------------------------------------------
    # NEW QUESTION
    # --------------------------------------------------

    question = st.chat_input(
        "Ask a question about the PDF"
    )


    if question:

        # Save user's question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)


        # Generate answer
        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching the document..."
                ):

                    retrieved_chunks = retrieve_chunks(
                        question,
                        chunks,
                        embeddings,
                        bm25,
                        embedding_model,
                        reranker,
                    )

                    answer = generate_answer(
                        question,
                        retrieved_chunks,
                    )

                st.write(answer)

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


                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_chunks,
                    }
                )


            except Exception:

                error_message = (
                    "The AI service is temporarily busy. "
                    "Please try your question again."
                )

                st.error(error_message)
