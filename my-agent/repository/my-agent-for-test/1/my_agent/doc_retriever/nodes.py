from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from .tools import get_docrag_tool
from ..state import AgentState


system_prompt = """
You are an expert document retrieval assistant.

Your responsibilities:
- Understand the user's request and break it down into clear, actionable steps if needed.
- Use the `document_retriever` tool to perform a semantic search for relevant documents.
- Carefully analyze the search results and return only the most relevant documents in a clear, structured format (e.g., with title, summary, and source).
- If no relevant documents are found, politely inform the user and suggest clarifying or refining their request if possible.
- If the retry count reaches 3 and no relevant documents are found, return an error message indicating the search was unsuccessful and recommend next steps.
- Only return the content of the documents that are relevant to the user's request, do not include any irrelevant information.

Remember: Your goal is to help the user efficiently find the information they need from the available documents.
"""


def doc_retriever_node(state: AgentState) -> dict:
    """
    Node to execute Python code using the PyCoder tool.
    """
    api_key = state.get("api_key", "")

    # Get the document retrieval tool
    docrag_tool = get_docrag_tool(api_key)

    # Create model for calling the tool
    model = ChatOpenAI(
        base_url="http://192.168.1.201/v1",
        model_name="meta/llama-3.1-8b-instruct",
        api_key=api_key,
        # max_completion_tokens=1024,
        temperature=0.9,
        top_p=0.9,
    ).bind_tools([docrag_tool])

    # Prepare the messages for the model
    # Extract non-system messages from state["messages"]
    messages = [msg for msg in state.get("messages", [])
                if not isinstance(msg, SystemMessage)]

    # create system message
    system_msg = SystemMessage(content=system_prompt + f"\n\nCurrent retry count: {state.get('retry_count', 0)}")
    messages.insert(0, system_msg)

    # If the last message is an AIMessage with content, add a user message to improve or execute the code
    last_message = messages[-1]
    if last_message and not isinstance(last_message, HumanMessage) and hasattr(last_message, 'content'):
        # Ask the user to execute the Python code
        user_msg = HumanMessage(
            content="Please search the document and provide the necessary information or context to address the request."
        )
        # Add the user message to the messages
        messages.append(user_msg)

    # Invoke the model with the messages
    res: BaseMessage = model.invoke(messages)
    print(f"Response from model: {res.content}")

    # return state with the response
    state["messages"].append(res)

    # Return the updated state
    return state


def should_continue(state:AgentState):
    """
    Function to determine whether to continue or end the workflow based on the last message's tool calls.
    """
    messages = state.get("messages", [])
    last_message = messages[-1]
    if not hasattr(last_message, 'tool_calls'):
        return "end"
    elif isinstance(last_message.tool_calls, list) and len(last_message.tool_calls) == 0:
        return "end"
    else:
        return "continue"
