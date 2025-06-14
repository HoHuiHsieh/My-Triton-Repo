from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from .nodes import doc_retriever_node, should_continue
from .tools import get_docrag_tool
from .state import AgentState


def get_doc_retriever_graph(api_key: str = "") -> CompiledStateGraph:

    tool_node = ToolNode([get_docrag_tool(api_key)])

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("doc_retriever_agent", doc_retriever_node)
    workflow.add_node("doc_retriever_action", tool_node)

    # Set the entrypoint as `agent`
    workflow.set_entry_point("doc_retriever_agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        "doc_retriever_agent",
        should_continue,
        {
            "continue": "doc_retriever_action",
            "end": END,
        },
    )

    # We now add a normal edge from `action` to `agent`.
    workflow.add_edge("doc_retriever_action", "doc_retriever_agent")

    # Finally, we compile the graph to create a runnable workflow
    graph = workflow.compile()

    # Return the compiled graph
    return graph
