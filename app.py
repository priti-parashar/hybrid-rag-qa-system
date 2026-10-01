import io
import re

import numpy as np
import pypdf
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)


# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG PDF Q&A",
    page_icon="📄",
)

st.title("📄 Hybrid RAG PDF Q&A")
st.caption(
    "Semantic Search + BM25 + CrossEncoder + Extractive QA"
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

    qa_model_name = (
        "distilbert/"
        "distilbert-base-uncased-distilled-squad"
    )

    qa_tokenizer = AutoTokenizer.from_pretrained(
        qa_model_name
    )

    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        qa_model_name
    )

    qa_model.eval()

    return (
        embedding_model,
        reranker,
        qa_tokenizer,
        qa_model,
    )


# --------------------------------------------------
# CLEAN PDF TEXT
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
# CHUNKING
# --------------------------------------------------

def chunk_text(
    text,
    words_per_chunk=140,
    overlap=35,
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

    reader = pypdf.PdfReader(
        io.BytesIO(file_bytes)
    )

    pages = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    full_text = "\n\n".join(pages)

    full_text = clean_text(
        full_text
    )

    chunks = chunk_text(
        full_text
    )

    if not chunks:
        return [], np.array([]), None

    (
        embedding_model,
        _,
        _,
        _,
    ) = load_models()

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

    return (
        chunks,
        embeddings,
        bm25,
    )


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
    top_k=5,
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
        bm25.get_scores(
            query_tokens
        )
    )

    semantic_scores = normalize_scores(
        semantic_scores
    )

    bm25_scores = normalize_scores(
        bm25_scores
    )

    # Hybrid retrieval
    hybrid_scores = (
        0.6 * semantic_scores
        + 0.4 * bm25_scores
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
        reranker.predict(
            pairs
        )
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

    return [
        chunks[index]
        for index in final_indices
    ]


# --------------------------------------------------
# EXTRACT ANSWER FROM ONE CHUNK
# --------------------------------------------------

def answer_from_chunk(
    question,
    context,
    tokenizer,
    model,
):

    import torch

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    start_logits = (
        outputs.start_logits[0]
    )

    end_logits = (
        outputs.end_logits[0]
    )

    sequence_ids = inputs.sequence_ids(
        0
    )

    context_positions = [
        index
        for index, sequence_id
        in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    if not context_positions:
        return "", 0.0

    best_answer = ""
    best_score = float("-inf")

    # Search possible answer spans only
    # inside the document context.
    for start_index in context_positions:

        max_end = min(
            start_index + 30,
            context_positions[-1] + 1,
        )

        for end_index in range(
            start_index,
            max_end,
        ):

            if sequence_ids[
                end_index
            ] != 1:
                continue

            score = float(
                start_logits[start_index]
                + end_logits[end_index]
            )

            if score > best_score:

                token_ids = inputs[
                    "input_ids"
                ][0][
                    start_index:
                    end_index + 1
                ]

                answer = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                ).strip()

                if answer:

                    best_score = score
                    best_answer = answer

    return (
        best_answer,
        best_score,
    )


# --------------------------------------------------
# FIND BEST ANSWER ACROSS RETRIEVED CHUNKS
# --------------------------------------------------

def find_best_answer(
    question,
    retrieved_chunks,
    tokenizer,
    model,
):

    answers = []

    for chunk in retrieved_chunks:

        answer, score = answer_from_chunk(
            question,
            chunk,
            tokenizer,
            model,
        )

        if answer:

            answers.append(
                (
                    answer,
                    score,
                    chunk,
                )
            )

    if not answers:

        return (
            None,
            None,
        )

    answers.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    best_answer = answers[0][0]
    best_source = answers[0][2]

    return (
        best_answer,
        best_source,
    )


# --------------------------------------------------
# USER INTERFACE
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
                qa_tokenizer,
                qa_model,
            ) = load_models()

        if not chunks:

            st.error(
                "No readable text was found "
                "in this PDF. It may be "
                "scanned or image-only."
            )

            st.stop()

        st.success(
            f"Ready! Extracted "
            f"{len(chunks)} chunks."
        )

    except Exception as error:

        st.error(
            "Something went wrong while "
            f"processing the PDF: {error}"
        )

        st.stop()


    question = st.text_input(
        "Ask a question about the PDF:"
    )


    if question:

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

                (
                    answer,
                    answer_source,
                ) = find_best_answer(
                    question,
                    retrieved_chunks,
                    qa_tokenizer,
                    qa_model,
                )


            st.subheader(
                "Answer"
            )

            if answer:

                st.write(
                    answer
                )

            else:

                st.warning(
                    "I couldn't find a reliable "
                    "answer in the document."
                )


            with st.expander(
                "Show source context"
            ):

                if answer_source:

                    st.markdown(
                        "**Answer source**"
                    )

                    st.write(
                        answer_source
                    )

                    st.divider()

                for number, chunk in enumerate(
                    retrieved_chunks[:3],
                    start=1,
                ):

                    st.markdown(
                        f"**Retrieved source "
                        f"{number}**"
                    )

                    st.write(
                        chunk
                    )

                    if number < min(
                        3,
                        len(retrieved_chunks),
                    ):

                        st.divider()


        except Exception as error:

            st.error(
                "Something went wrong while "
                f"answering: {error}"
            )
