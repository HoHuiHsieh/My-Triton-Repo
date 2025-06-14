from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from .tools import get_pycoder_tool
from ..state import AgentState


pycoder_tool = get_pycoder_tool()
tool_node = ToolNode([pycoder_tool])

system_prompt = """
You are an expert Python code execution assistant.
Your primary task is to execute Python code provided by the user and return the result.
Always use the `python_coder` tool to execute the code.

Instructions:
- The code you execute must assign the output to a variable named 'result'. Only the value of 'result' will be returned.
- If an error occurs during execution, return the error message.
- Do not explain the code unless explicitly asked.
- Only execute safe, non-destructive code. Do not perform file I/O, network requests, or system modifications.

Example usage:
```python
# Calculate a mathematical expression
result = 2 + 2

# Generate a list of numbers
numbers = [i for i in range(10)]
result = sum(numbers)

# Process text
text = "hello world"
result = text.upper()
```
"""


def pycoder_node(state: AgentState) -> dict:
    """
    Node to execute Python code using the PyCoder tool.
    """
    # Create model for calling the tool
    model = ChatOpenAI(
        base_url="http://192.168.1.201/v1",
        model_name="meta/llama-3.1-8b-instruct",
        api_key=state.get("api_key", ""),
        max_completion_tokens=1024,
        temperature=0.9,
        top_p=0.9,
        n=1,
    )
    # Bind tools to the model
    model = model.bind_tools(
        [pycoder_tool],
        strict=True
    )
    model = model.bind(response_format={"type": "json_object"})

    # Prepare the messages for the model
    # Extract non-system messages from state["messages"]
    messages = [msg for msg in state.get(
        "messages", []) if not isinstance(msg, SystemMessage)]

    # create system message
    system_msg = SystemMessage(content=system_prompt)
    messages.insert(0, system_msg)

    # If the last message is an AIMessage with content, add a user message to improve or execute the code
    last_message = messages[-1]
    if last_message and isinstance(last_message, AIMessage) and hasattr(last_message, 'content'):
        last_content = last_message.content
        if 'Python execution error' in last_content:
            # Ask the user to improve the code to avoid the execution error
            user_msg = HumanMessage(
                content="Improve the code to avoid the execution error.")
        else:
            # Ask the user to execute the Python code
            user_msg = HumanMessage(
                content="Solve the problem in the last message.")

        # Add the user message to the messages
        messages.append(user_msg)

    # Invoke the model with the messages
    res: BaseMessage = model.invoke(messages)

    # return state with the response
    state["messages"].append(res)

    # Return the updated state
    return state


def should_continue(state: AgentState):
    """
    Function to determine whether to continue or end the workflow based on the last message's tool calls.
    """
    messages = state.get("messages", [])
    last_message = messages[-1]
    last_content = last_message.content if hasattr(
        last_message, 'content') else ""
    if 'Python execution error' in last_content:
        return "continue"
    else:
        return "end"
