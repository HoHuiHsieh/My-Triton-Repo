from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from .state import AgentState


@lru_cache(maxsize=4)
def _get_model(openai_apikey: str = None):
    model = ChatOpenAI(
        model_name="gpt-4o-mini-2024-07-18",
        api_key=openai_apikey,
        max_completion_tokens=1024,
        temperature=0.9,
        top_p=0.9,
        n=1,
    )
    return model


system_prompt = """
You are an expert assistant. Your task is to answer the user's request based on previous messages in the conversation.
You should provide a clear and concise response to the user's request.
Do not include any additional information or context that is not directly related to the user's request.
Do not include any harmful, unethical, or illegal content in your response.
If you are unsure about the user's request, ask for clarification.
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
