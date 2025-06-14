from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from .state import AgentState


system_prompt = """
You are an expert AI assistant skilled in step-by-step reasoning and problem solving.
When given a user question or task, you always think out loud, breaking down your reasoning into clear, logical steps before acting.
You will use the following format to respond:

<think>
A step of reasoning or thought process. Explain your assumptions, logic, and any calculations you perform.
</think>
<think>
Next step or action to take based on your reasoning.
</think>
...etc.

You should always use the following 4 tools in your actions:
1. 'document_retrieval': For retrieving relevant documents or information, including:
    - Movie reviews
2. 'python_coder': For executing Python code to perform calculations or data processing, including:
    - Calculate a mathematical expression
    - Generate a list of numbers
    - Perform data analysis
    - Generate a random number
    - Process text
    - Retrieve system clock time
    - Other Python-related tasks
3. 'replan': If you need to change your plan based on new information or insights, including:
    - Adjusting your problem-solving approach
    - Reassessing your assumptions
4. 'end': If you have completed the task and no further action is needed, including:
    - Ending the workflow when the task is complete
    - Ending the workflow when no further action is required
    - Ending the workflow when the task or question has been answered or resolved
    - Ending the workflow when the task or question cannot be answered or resolved

Your response should include the full name of the tool you are invoking, such as 'document_retrieval', 'python_coder', 'replan', or 'end'.
Explain your reasoning before invoking any tool.

Only reason and plan based on the user's input and previous messages.
Now, let's start solving the user's problem step by step.
"""


def planer_node(state: AgentState) -> dict:
    """
    Node to plan the next steps based on the user's input and previous messages.
    """
    # Create model for calling the tool
    model = ChatOpenAI(
        model_name="gpt-4o-mini-2024-07-18",
        api_key=state.get("api_key", ""),
        max_completion_tokens=1024,
        temperature=0.9,
        top_p=0.9,
        n=1,
        stop_sequences=["</think>"],  # Stop on think tags
    )

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


def guardrail(state: AgentState):
    """
    Function to determine whether to continue or end the workflow based on the last message's tool calls.
    
    """
    messages = state.get("messages", [])
    last_message = messages[-1]
    last_content = last_message.content if hasattr(
        last_message, 'content') else ""

    # find index of the tool in the last message
    tools = ["document_retrieval", "python_coder", "replan", "end"]
    indexes = [last_content.find(tool) for tool in tools]
    # sort indexes to find the first tool mentioned
    first_tool_index = min((index for index in indexes if index != -1), default=-1)
    if first_tool_index == -1:
        # If no tool is mentioned, replan the workflow
        return "replan"
    # Determine which tool was mentioned first
    first_tool = tools[indexes.index(first_tool_index)]
    if first_tool == "document_retrieval":
        return "document_retrieval"
    elif first_tool == "python_coder":
        return "python_coder"
    elif first_tool == "replan":
        return "replan"
    elif first_tool == "end":
        return "end"
