# Hybrid RAG PDF Q&A

A deployed PDF question-answering application built with Python and Streamlit. The system uses hybrid retrieval, combining semantic search and BM25, followed by CrossEncoder reranking and Gemini for grounded answer generation.

Users can upload a text-based PDF and ask questions about its contents through a chat-style interface.

## Live Demo

https://hybrid-rag-app-system-thid7wibnjesxfzgwuzbkd.streamlit.app/

## Features

- Upload and process PDF documents
- Extract and clean PDF text
- Split documents into overlapping chunks
- Semantic retrieval using sentence embeddings
- Keyword retrieval using BM25
- Hybrid ranking combining semantic and BM25 scores
- CrossEncoder reranking for improved relevance
- Gemini-based grounded answer generation
- Chat-style question-answering interface
- Display retrieved source context for transparency
- Automatic retry and model fallback for temporary API errors
- Secure API-key handling using Streamlit Secrets

## Architecture

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Overlapping Chunking
    ↓
┌───────────────────────┐
│                       │
Semantic Search       BM25 Search
│                       │
└───────────┬───────────┘
            ↓
     Hybrid Retrieval
            ↓
   CrossEncoder Reranking
            ↓
     Top Relevant Chunks
            ↓
        Gemini API
            ↓
     Grounded Answer
            ↓
 Answer + Source Context
```

## Tech Stack

- Python
- Streamlit
- pypdf
- NumPy
- Sentence Transformers
- `all-MiniLM-L6-v2`
- BM25 (`rank-bm25`)
- CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- Google Gemini API
- Git & GitHub
- Streamlit Community Cloud

## How It Works

### 1. PDF Processing

The uploaded PDF is read using `pypdf`. Extracted text is cleaned and divided into overlapping chunks.

### 2. Semantic Search

Each chunk is converted into a vector embedding using the `all-MiniLM-L6-v2` Sentence Transformer model.

The user's question is also converted into an embedding, allowing the system to find semantically similar chunks.

### 3. BM25 Search

BM25 performs keyword-based retrieval and helps identify chunks containing important terms from the user's question.

### 4. Hybrid Retrieval

Semantic similarity and BM25 scores are normalized and combined to produce a hybrid retrieval score.

### 5. CrossEncoder Reranking

The strongest candidate chunks are reranked using a CrossEncoder model to improve relevance before answer generation.

### 6. Grounded Answer Generation

Only the top retrieved document chunks are sent to Gemini along with the user's question.

Gemini is instructed to answer using only the supplied document context and to avoid inventing information that is not present in the retrieved text.

## Installation

Clone the repository:

```bash
git clone https://github.com/priti-parashar/hybrid-rag-qa-system.git
cd hybrid-rag-qa-system
```

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## API Key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Do not commit this file or your API key to GitHub.

## Run Locally

```bash
streamlit run app.py
```

Then upload a PDF and ask questions about the document.

## Project Structure

```text
hybrid-rag-qa-system/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- Designed primarily for text-based PDFs.
- Scanned or image-only PDFs require OCR, which is not currently implemented.
- Answer quality depends on successful retrieval of relevant document chunks.
- Response time and availability can be affected by external model API availability.
- Uploaded document context used for answer generation is sent to an external LLM API.

## Future Improvements

- OCR support for scanned PDFs
- Page-level source citations
- Improved retrieval evaluation
- Support for multiple documents
- Configurable retrieval parameters
- More robust conversation-aware follow-up questions

## Author

**Preeti Parashar**

Junior Python Developer | AI/ML Enthusiast
