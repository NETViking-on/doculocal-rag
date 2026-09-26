# DocuLocal-RAG

> A fully local, privacy-first RAG (Retrieval-Augmented Generation) system for chatting with your own documents — no data ever leaves your machine.

## Problem

Cloud-based AI tools (ChatGPT, Claude, etc.) are great for Q&A over documents, but they require uploading potentially sensitive data to third-party servers. DocuLocal-RAG solves this by running the entire pipeline — embeddings, vector search, and the LLM itself — locally on consumer hardware.

## Who it's for

- Anyone who wants to query personal notes, PDFs, or research papers without cloud dependency
- A learning project demonstrating production-grade RAG engineering: chunking, embeddings, hybrid retrieval, and evaluation

## Features (v1 scope)

- [ ] Ingest PDF and TXT documents
- [ ] Chunking + local embedding generation
- [ ] Vector storage and semantic search (ChromaDB)
- [ ] Question answering with **source citation** (file name + page/paragraph)
- [ ] Simple CLI interface

### Explicitly out of scope for v1
- Multi-agent orchestration
- Fine-tuning custom models
- Polished web UI (CLI first, FastAPI later)
- Multi-user support / authentication

## Tech Stack

| Component | Tool |
|---|---|
| LLM inference | [Ollama](https://ollama.com) (local, runs on RTX 4060 Ti 16GB) |
| Embeddings | Local embedding model via Ollama or `sentence-transformers` |
| Vector DB | [ChromaDB](https://www.trychroma.com/) |
| Document parsing | `pypdf`, `python-docx` |
| Orchestration | Python, [LangChain](https://www.langchain.com/) |
| API layer (later) | FastAPI |
| Evaluation (later) | [RAGAS](https://github.com/explodinggradients/ragas) |

**Hardware:** Developed and tested on RTX 4060 Ti 16GB VRAM, 32GB RAM, running locally via Ollama.

## Roadmap / Milestones

1. **M1 — Core pipeline (Jupyter notebook):** load PDFs → chunk → generate embeddings → store in ChromaDB
2. **M2 — CLI Q&A:** ask a question, retrieve relevant chunks, generate a grounded answer with the local LLM
3. **M3 — Source citation:** every answer references the exact document and location it came from
4. **M4 — Hybrid search:** combine semantic (embedding) search with keyword (BM25) search for better retrieval
5. **M5 — Evaluation:** measure retrieval quality with RAGAS metrics (precision, recall, faithfulness)
6. **M6 — Deployment:** wrap in a FastAPI endpoint + minimal web UI

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/doculocal-rag.git
cd doculocal-rag

# Install dependencies
pip install -r requirements.txt

# Pull a local model via Ollama
ollama pull llama3.1:8b

# Run ingestion
python ingest.py --docs-dir ./documents

# Ask a question
python query.py "What does this document say about X?"
```

*(Setup instructions will be finalized as the project develops.)*

## Why this project

This project was built to learn production RAG engineering hands-on: chunking strategy, embedding model selection, retrieval pipeline design, and evaluation — the core skills behind real-world AI engineering roles — while keeping full data privacy through local-only inference.

## License

MIT