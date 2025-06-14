from langgraph.graph import StateGraph, END
from .nodes import pycoder_node, tool_node, should_continue
from ..state import AgentState


# Define a new graph
workflow = StateGraph(AgentState)


# Define the two nodes we will cycle between
workflow.add_node("pycoder_agent", pycoder_node)
workflow.add_node("pycoder_action", tool_node)

# Set the entrypoint as `pycoder_agent`
workflow.set_entry_point("pycoder_agent")

# We now add a normal edge from `agent` to `action`.
workflow.add_edge("pycoder_agent", "pycoder_action")

# We now add a conditional edge
workflow.add_conditional_edges(
    "pycoder_action",
    should_continue,
    {
        "continue": "pycoder_agent",
        "end": END,
    },
)



# Finally, we compile the graph to create a runnable workflow
graph = workflow.compile()
