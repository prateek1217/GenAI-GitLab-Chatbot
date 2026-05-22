import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

SCRAPED_FILE = "backend/scraped_data.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension


def load_scraped_docs(filepath: str) -> list[Document]:
    with open(filepath, encoding="utf-8") as f:
        raw_docs = json.load(f)

    docs = []
    for page in raw_docs:
        content = page.get("markdown", "").strip()
        if not content or len(content) < 200:
            continue

        metadata = page.get("metadata", {})
        docs.append(Document(
            page_content=content,
            metadata={
                "source": page.get("url", ""),
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
            },
        ))

    print(f"Loaded {len(docs)} documents from {filepath}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def setup_pinecone_index(pc: Pinecone, index_name: str):
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{index_name}' already exists.")
    return pc.Index(index_name)


def ingest(scraped_file: str = SCRAPED_FILE):
    docs = load_scraped_docs(scraped_file)
    chunks = chunk_documents(docs)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("INDEX_NAME", "question-answer")
    index = setup_pinecone_index(pc, index_name)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    print(f"Ingesting {len(chunks)} chunks into Pinecone in batches of {BATCH_SIZE}...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"  Progress: {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("\nIngestion complete!")


if __name__ == "__main__":
    ingest()
