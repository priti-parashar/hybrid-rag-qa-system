import pypdf
import re
import pickle
import os

EMBEDDINGS_CACHE = "embeddings_cache.pkl"

pdf = pypdf.PdfReader("sample.pdf")

text = ""
for page in pdf.pages:
    text += page.extract_text() + "\n"

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

chunks = chunk_by_definition(text)
if not chunks:
    print("No 'Term/Definition/Example' structure found — using sentence-based chunking instead.")
    chunks = chunk_by_sentences(text)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(EMBEDDINGS_CACHE):
    with open(EMBEDDINGS_CACHE, "rb") as f:
        cached_chunks, embeddings = pickle.load(f)
    if cached_chunks != chunks:
        embeddings = model.encode(chunks)
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump((chunks, embeddings), f)
else:
    embeddings = model.encode(chunks)
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump((chunks, embeddings), f)

from sentence_transformers import util
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import pipeline

tokenized_chunks = [re.findall(r'\w+\b', chunk.lower()) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

CONFIDENCE_THRESHOLD = 0.1

while True:
    query = input("\nAsk a question(or type 'exit' to quit): ")
    if query.strip().lower() in ("exit", "quit"):
        print("Goodbye!")
        break

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
    best_rerank_index = int(top_indices[best_position])

    context = chunks[best_rerank_index]

    result = qa(question=query, context=context)

    print("\nAnswer:")
    if result['score'] < CONFIDENCE_THRESHOLD:
        print("I'm not confident enough to answer this from the document. Try rephrasing your question.")
    else:
        print(result['answer'])

    print("\nConfidence:")
    print(result['score'])