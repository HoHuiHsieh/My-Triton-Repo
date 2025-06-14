from functools import lru_cache
from langchain_core.tools import tool
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field


@lru_cache(maxsize=4)
def _get_vectorstore(api_key: str = None):
    """Get a vector store for document retrieval."""
    # Prepare the vector store and embedding
    connection = "postgresql+psycopg://postgresql:password@postgres:5432/postgres"
    collection = "document_collection"
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        check_embedding_ctx_length=False,
    )
    # Return the vector store
    return PGVector(
        embeddings=embedding,
        connection=connection,
        collection_name=collection,
        use_jsonb=True,
    )


class RetrieverInputSchema(BaseModel):
    """Input schema for the document retriever tool."""
    query: str = Field(description="The search query used to find relevant documents.")
    num: int = Field(description="The number of documents to retrieve. Defaults to 1.")


def get_docrag_tool(openai_apikey: str = None):
    """Get a retriever tool function."""
    vector_store = _get_vectorstore(openai_apikey)

    @tool("document_retriever", args_schema=RetrieverInputSchema, response_format="content_and_artifact")
    def document_retriever(query: str, num: int = 1) -> tuple[str, list]:
        """
        Call this tool to retrieve relevant documents from a vector store based on a semantic similarity search.
        """
        retrieved_docs = vector_store.similarity_search(query, k=num)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        serialized = "\n```tool\nRetrieved documents:\n\n" + serialized + "\n```\n"
        return serialized, retrieved_docs

    return document_retriever
