from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage
from typing import TypedDict, Literal, Union, Callable, List
from langgraph.graph import StateGraph, END
from .doc_retriever import get_doc_retriever_graph
from .py_coder import graph as py_coder_graph
from .planner import guardrail, planer_node
from .summarize import summarize_node
from .state import AgentState


class RawMessage(TypedDict):
    role: Literal["user", "assistant", "tool"]
    content: str


async def run_graph(input: List[RawMessage], openai_apikey: str, callback: Callable):
    """
    Run the graph asynchronously with the given input and OpenAI API key.
    The input should be a list of AnyMessage objects, and the callback will be called with the response content.
    """
    # Convert the input to a list of HumanMessage, AIMessage, ToolMessage objects
    messages: List[AnyMessage] = []
    for msg in input:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "tool":
            messages.append(ToolMessage(content=msg["content"]))
        else:
            pass  # Ignore any other roles

    # Create the initial state with messages and OpenAI API key
    initial_state: AgentState = {
        "messages": messages,
        "api_key": openai_apikey,
    }

    # Define a new graph
    workflow = StateGraph(AgentState)
    doc_retriever_graph = get_doc_retriever_graph(openai_apikey)

    # Define the two nodes we will cycle between
    workflow.add_node("planer", planer_node)
    workflow.add_node("document_retrieval", doc_retriever_graph)
    workflow.add_node("python_coder", py_coder_graph)
    workflow.add_node("summarize", summarize_node)

    # Set the entrypoint as `agent`
    workflow.set_entry_point("planer")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        "planer",
        guardrail,
        {
            "document_retrieval": "document_retrieval",
            "python_coder": "python_coder",
            "replan": "planer",
            "end": "summarize",
        },
    )

    # We now add a normal edges
    workflow.add_edge("document_retrieval", "planer")
    workflow.add_edge("python_coder", "planer")
    workflow.add_edge("summarize", END)

    # Finally, we compile the graph to create a runnable workflow
    graph = workflow.compile()

    # Start thinking by sending the initial input to the callback
    callback("<think>")

    # Start the graph execution with the input
    async for output in graph.astream(initial_state, stream_mode="updates"):
        for key, value in output.items():
            # get the last message from the output
            msg: Union[AIMessage, ToolMessage] = value["messages"][-1]
            content = msg.content if hasattr(msg, 'content') else str(msg)
            # process the content based on the message type
            if key in ["python_coder", "document_retrieval"]:
                # send the content to the callback
                callback(content)
            elif key == "planer":
                # send the content to the callback with a think tag
                if "<think>" in content and "</think>" in content:
                    #  Get the content between <think> and </think>
                    content = content.split("<think>")[-1].split("</think>")[0]
                elif "</think>" in content:
                    #  Get the content before </think>
                    content = content.split("</think>")[0]
                elif "<think>" in content:
                    #  Get the content after <think>
                    content = content.split("<think>")[-1]
                # send the content to the callback
                callback(content)
            elif key == "summarize":
                callback(f"</think>\n\n{content}")
            else:
                pass  # Ignore any other keys
