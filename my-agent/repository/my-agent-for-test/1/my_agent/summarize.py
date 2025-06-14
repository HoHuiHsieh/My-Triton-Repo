from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from .state import AgentState


@lru_cache(maxsize=4)
def _get_model(openai_apikey: str = None):
    model = ChatOpenAI(
        base_url="http://192.168.1.201/v1",
        model_name="meta/llama-3.1-8b-instruct",
        api_key=openai_apikey,
        max_completion_tokens=1024,
        temperature=0.9,
        top_p=0.9,
        n=1,
    )
    return model


system_prompt = """
You are an expert assistant.
Summarize the conversation so far in clear, concise language.
Focus on the main points, user requests, and any important context.
Exclude unnecessary details and keep the summary brief and informative.
"""


def summarize_node(state: AgentState) -> dict:
    """
    Node to answer user's request based on the user's input and previous messages.
    """
    # Create model for calling the tool
    model = _get_model(state.get("api_key", ""))

    # Prepare the messages for the model
    # Extract non-system messages from state["messages"]
    messages = [msg for msg in state.get("messages", [])
                if not isinstance(msg, SystemMessage)]

    # create system message
    system_msg = SystemMessage(content=system_prompt)
    messages.insert(0, system_msg)

    # Invoke the model with the messages
    res: BaseMessage = model.invoke(messages)

    # return state with the response
    state["messages"].append(res)

    # Return the updated state
    return state
