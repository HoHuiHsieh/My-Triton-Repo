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

If you need to use tools or ask clarifying questions, explain your reasoning in the <think> section first.
Always be explicit about your assumptions and logic.
Do not skip steps, and avoid making unsupported leaps in reasoning.
If you are unsure, state your uncertainty and suggest what information would help.

Here are some tools that you can use in your actions:
- 'document_retrieval': For retrieving relevant documents or information.
- 'python_coder': For executing Python code to perform calculations or data processing.

Now, let's start solving the user's problem step by step.
"""


def planer_node(state: AgentState) -> dict:
    """
    Node to plan the next steps based on the user's input and previous messages.
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


guardrail_prompt = """
The plan you just created:
```
{{ plan }}
```

Based on the plan, you need to decide which tool to use next. Here are your options:
- If the plan requires retrieving documents or information, use the `document_retrieval` tool.
- If the plan requires executing Python code, use the `python_coder` tool.
- If the plan does not require any further action, simply end the workflow.

Please choose one of the following options:
- `document_retrieval`: Use this tool to retrieve relevant documents or information.
- `python_coder`: Use this tool to execute Python code for calculations or data processing.
- `end`: End the workflow if no further action is needed.
"""


def guardrail(state: AgentState):
    """
    Function to determine whether to continue or end the workflow based on the last message's tool calls.
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
        stop_sequences=["</think>"],  # Stop on think tags
    )

    # Get the last AIMessage from the state
    messages = state.get("messages", [])
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        # If the last message is an AIMessage with content, use it to create the prompt
        last_content = last_message.content if hasattr(
            last_message, 'content') else ""
        prompt = guardrail_prompt.replace("{{ plan }}", last_content)

        # Create a user message with the prompt, which will be used to invoke the model
        response = model.invoke([HumanMessage(content=prompt)])

        # Route based on the response content
        response_content = response.content if hasattr(
            response, 'content') else ""
        if 'document_retrieval' in response_content:
            return "document_retrieval"
        elif 'python_coder' in response_content:
            return "python_coder"
        else:
            return "end"
