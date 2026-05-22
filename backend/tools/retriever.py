import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import BaseTool
from pinecone import Pinecone

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_retriever_tool() -> BaseTool:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("INDEX_NAME", "question-answer"))
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    return create_retriever_tool(
        retriever,
        name="search_gitlab_handbook",
        description=(
            "Search GitLab's Handbook and Direction pages. Use this for questions about "
            "GitLab's values, culture, remote work policies, engineering practices, "
            "hiring process, product direction, company guidelines, and internal processes."
        ),
    )
