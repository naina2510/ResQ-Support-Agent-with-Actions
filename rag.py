"""
Builds the vector index used for retrieval.

Reads the knowledge base markdown, splits it into chunks, embeds them with
Gemini, and stores them in FAISS. Returns a retriever for the agent's
knowledge base tool.

Gemini embeddings and FAISS were picked to keep the install small: no torch,
no sentence-transformers, no chromadb server to run.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.md")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBED_MODEL = "models/gemini-embedding-001"


def _load_and_split():
    """Read the knowledge base and split it into overlapping chunks.

    Splitting on markdown headings first keeps each chunk on a single policy
    topic. The size cap stops chunks getting too broad to retrieve usefully.
    """
    with open(KB_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        chunk_size=600,
        chunk_overlap=80,
    )
    return splitter.create_documents([text])


def get_retriever(api_key: str, k: int = 3):
    """Return a retriever over the knowledge base.

    The index is built on the first run and saved to disk, so restarts skip
    the embedding step. Delete the faiss_index folder to force a rebuild
    after editing the knowledge base.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=api_key
    )

    index_file = os.path.join(PERSIST_DIR, "index.faiss")
    if os.path.exists(index_file):
        vectordb = FAISS.load_local(
            PERSIST_DIR,
            embeddings,
            allow_dangerous_deserialization=True,  # index is our own file
        )
    else:
        docs = _load_and_split()
        vectordb = FAISS.from_documents(docs, embeddings)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        vectordb.save_local(PERSIST_DIR)

    return vectordb.as_retriever(search_kwargs={"k": k})
