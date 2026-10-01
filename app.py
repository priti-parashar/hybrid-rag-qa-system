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
st.caption(
    "Semantic Search + BM25 + CrossEncoder Reranking"
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

def chunk_text(text, words_per_chunk=180, overlap=40):

    words = text.split()

    chunks = []

    if not words:
        return chunks

    step = words_per_chunk - overlap

    for start in range(0, len(words), step):

        chunk_words = words[
            start:start + words_per_chunk
        ]

        if not chunk_words:
            continue

        chunk = " ".join(chunk_words).strip()

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

    pdf = pypdf.PdfReader(file_bytes)

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
# NORMALIZATION
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

    # Semantic retrieval
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    semantic_scores = util.cos_sim(
        query_embedding,
        embeddings
    )[0].cpu().numpy()


    # BM25 retrieval
    query_tokens = re.findall(
        r"\b\w+\b",
        query.lower()
    )

    bm25_scores = np.asarray(
        bm25.get_scores(query_tokens)
    )


    # Normalize both score types
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

        for position in reranked_positions[:top_k]
    ]


    final_chunks = [

        chunks[index]

        for index in final_indices
    ]


    final_scores = [

        float(
            rerank_scores[position]
        )

        for position in reranked_positions[:top_k]
    ]


    return final_chunks, final_scores


# --------------------------------------------------
# SENTENCE EXTRACTION
# --------------------------------------------------

def split_into_sentences(text):

    # Split normal sentences while also handling
    # PDF text that contains headings or short lines.
    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    return [

        sentence.strip()

        for sentence in sentences

        if sentence.strip()
    ]


# --------------------------------------------------
# CREATE ANSWER FROM RETRIEVED CONTEXT
# --------------------------------------------------

def create_answer(
    query,
    retrieved_chunks,
    embedding_model
):

    sentences = []

    for chunk in retrieved_chunks:

        sentences.extend(
            split_into_sentences(chunk)
        )


    # Remove duplicate sentences
    unique_sentences = []

    seen = set()

    for sentence in sentences:

        normalized = sentence.lower().strip()

        if (
            normalized
            and normalized not in seen
        ):

            seen.add(normalized)

            unique_sentences.append(
                sentence
            )


    if not unique_sentences:

        return None, 0.0


    # Compare the question directly with
    # sentences in the retrieved context.
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sentence_embeddings = embedding_model.encode(
        unique_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(
        query_embedding,
        sentence_embeddings
    )[0].cpu().numpy()


    best_indices = np.argsort(
        similarities
    )[::-1]


    best_index = int(
        best_indices[0]
    )

    best_score = float(
        similarities[best_index]
    )

    best_sentence = unique_sentences[
        best_index
    ]


    # --------------------------------------------------
    # SPECIAL HANDLING FOR COMPLEXITY QUESTIONS
    # --------------------------------------------------

    query_lower = query.lower()

    if (
        "complexity" in query_lower
        or "big o" in query_lower
        or "big-o" in query_lower
    ):

        complexity_pattern = re.compile(
            r"O\s*\(\s*[^)]+\s*\)",
            re.IGNORECASE
        )

        query_words = set(
            re.findall(
                r"\b[a-zA-Z]+\b",
                query_lower
            )
        )

        candidates = []

        for sentence in unique_sentences:

            complexities = complexity_pattern.findall(
                sentence
            )

            if not complexities:
                continue

            sentence_words = set(
                re.findall(
                    r"\b[a-zA-Z]+\b",
                    sentence.lower()
                )
            )

            overlap = len(
                query_words & sentence_words
            )

            candidates.append(
                (
                    overlap,
                    sentence
                )
            )


        if candidates:

            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

            complexity_sentence = candidates[0][1]

            return (
                complexity_sentence,
                best_score
            )


    # --------------------------------------------------
    # NORMAL ANSWER
    # --------------------------------------------------

    return best_sentence, best_score


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
                "The PDF may contain scanned images instead of text."
            )

            st.stop()


        st.success(
            f"Ready! Extracted {len(chunks)} chunks."
        )


    except Exception as error:

        st.error(
            f"Something went wrong while processing the PDF: {error}"
        )

        st.stop()


    # --------------------------------------------------
    # QUESTION
    # --------------------------------------------------

    query = st.text_input(
        "Ask a question about the PDF:"
    )


    if query:

        try:

            with st.spinner(
                "Searching the document..."
            ):

                retrieved_chunks, rerank_scores = retrieve_chunks(
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


            st.subheader(
                "Answer"
            )


            if (
                answer is None
                or confidence < 0.20
            ):

                st.warning(
                    "I couldn't find a reliable answer "
                    "to this question in the document."
                )

            else:

                st.write(
                    answer
                )


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

                    st.write(
                        chunk
                    )

                    if number < min(
                        3,
                        len(retrieved_chunks)
                    ):

                        st.divider()


        except Exception as error:

            st.error(
                f"Something went wrong while answering: {error}"
            )
