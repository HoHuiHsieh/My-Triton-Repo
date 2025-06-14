#!/usr/bin/env python3
"""
This is a demonstration script showing how to use the py_coder module.
"""
import sys
import os
sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), 
                        "/workspace/repository/my-agent-for-test/1/my_agent/"
                        )
                    )
                )
from doc_retriever import get_doc_retriever_graph
from doc_retriever.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from utils import set_variables


api_key = set_variables().get("API_KEY", "OPENAI_APIKEY")


# Prepare the vector store and embedding
connection = "postgresql+psycopg://postgresql:password@postgres:5432/postgres"
collection = "document_collection"
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]

def run_doc_retriever_demo():
    """
    Demonstrates how to use the py_coder module by running a simple example.
    """
    # You should set your API key here or in the environment
    api_key = set_variables().get("API_KEY", "OPENAI_APIKEY")
    
    # Create an initial state with an assistant message
    state = AgentState(
        messages=[
            HumanMessage(content="Find the year, name and director of the most relevant movie to dinosaurs."),
            AIMessage(content="Ok, I will find in document."), 
        ],
        api_key=api_key
    )
    graph = get_doc_retriever_graph(api_key)
    
    print("Initial state:", state)
    print("\nRunning the doc_retriever graph...\n")
    
    # Execute the graph
    try:
        final_state = graph.invoke(state)
        print("\nFinal state:")
        for msg in final_state["messages"]:
            print(f"\n{msg.type}: {msg.content}")

        print("\nDone! Check the final state for results.")
    except Exception as e:
        print(f"Error running the graph: {str(e)}")

if __name__ == "__main__":
    # embedding = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    #     api_key=api_key,
    #     check_embedding_ctx_length=False,
    # )
    # PGVector.from_documents(
    #     documents=docs,
    #     ids=[str(i) for i in range(len(docs))],
    #     embedding=embedding,
    #     connection=connection,
    #     collection_name=collection,
    #     pre_delete_collection=True,
    #     use_jsonb=True,
    # )
    # print("Vector store initialized with sample documents.")
    # print("===================")
    print("DocRetriver Module Demo")
    print("===================")
    run_doc_retriever_demo()
