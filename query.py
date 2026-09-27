"""
DocuLocal-RAG — Command-line Q&A interface.

Retrieves the most relevant chunks from the local ChromaDB collection
and asks a local LLM (via Ollama) to generate a grounded answer.

Usage:
    python query.py "What is the attention mechanism in Transformers?"
"""

import sys

import chromadb
import ollama
from chromadb.utils import embedding_functions

# --- Configuration -----------------------------------------------------------

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "doculocal_rag"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
TOP_K = 3  # number of chunks retrieved per query


# --- Core functions ------------------------------------------------------------

def load_collection():
    """Connect to the existing persistent ChromaDB collection.

    Raises a clear error if the database hasn't been populated yet,
    instead of failing deep inside ChromaDB's internals.
    """
    embedding_fn = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load collection '{COLLECTION_NAME}' from {CHROMA_DB_PATH}. "
            "Run the ingestion pipeline (rag_pipeline.ipynb) first."
        ) from exc


def retrieve_chunks(collection, question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k chunks most relevant to the question.

    Each result includes the chunk text plus its source file and page number,
    so citations can point to an exact location rather than just a filename.
    """
    results = collection.query(query_texts=[question], n_results=top_k)

    retrieved = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "text": text,
            "source": meta["source"],
            "page": meta.get("page"),  # may be absent for older collections
        })
    return retrieved


def format_citation(source: str, page: int | None) -> str:
    """Render a human-readable citation, with or without a page number."""
    if page is not None:
        return f"{source}, page {page}"
    return source


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """Assemble a grounded prompt from retrieved chunks and the question.

    Explicitly asks for a thorough answer — without this, the model tends
    to default to a one-line summary even when given rich context.
    """
    context_blocks = [
        f"[Source {i}: {format_citation(chunk['source'], chunk['page'])}]\n{chunk['text']}"
        for i, chunk in enumerate(retrieved, start=1)
    ]
    context = "\n\n".join(context_blocks)

    return f"""You are a helpful assistant answering questions using only the provided context.

Instructions:
- Answer thoroughly, using all relevant details found in the context.
- Explain key terms and reasoning, not just a one-line conclusion.
- If the answer is not contained in the context, say so explicitly — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(prompt: str) -> str:
    """Call the local LLM via Ollama to generate a grounded answer."""
    try:
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama or model '{LLM_MODEL}'. "
            "Make sure Ollama is running and the model is pulled."
        ) from exc
    return response["response"]


def ask(question: str) -> None:
    """Full RAG flow: retrieve relevant chunks, then generate an answer."""
    collection = load_collection()
    retrieved = retrieve_chunks(collection, question)

    if not retrieved:
        print("No relevant information found in the document collection.")
        return

    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt)

    print("\n--- Answer ---")
    print(answer)

    print("\n--- Sources ---")
    seen = set()
    for chunk in retrieved:
        citation = format_citation(chunk["source"], chunk["page"])
        if citation not in seen:
            print(f"- {citation}")
            seen.add(citation)


# --- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    user_question = " ".join(sys.argv[1:])

    try:
        ask(user_question)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)