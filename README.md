# PDF Q&A — Hybrid RAG System

Ask questions about any PDF and get accurate, source-grounded answers.

## How it works
1. **Chunking** — Splits the PDF into meaningful chunks. Uses structured Term/Definition/Example detection when available, falling back to sentence-based chunking for general documents.
2. **Hybrid retrieval** — Combines semantic search (sentence embeddings) with keyword search (BM25) to find the most relevant chunks.
3. **Reranking** — A cross-encoder reranks the top candidates for precision.
4. **Answer extraction** — A fine-tuned QA model (RoBERTa, SQuAD2) extracts the exact answer span from the best-matching chunk, with a confidence score.

## Tech stack
- `sentence-transformers` (embeddings + cross-encoder reranking)
- `rank-bm25` (keyword search)
- `transformers` (question-answering pipeline)
- `streamlit` (web UI)
- `pypdf` (PDF text extraction)

## Running it

**CLI version:**