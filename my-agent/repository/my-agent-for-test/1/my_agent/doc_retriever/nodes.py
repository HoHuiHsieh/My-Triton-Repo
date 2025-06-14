from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from .tools import get_docrag_tool
from .state import AgentState

system_prompt = """
You are an expert document retrieval assistant specializing in semantic search.

When searching for documents:
- Prioritize relevance and accuracy over quantity
- Extract and use key terms from the user's query
- Leverage synonyms and related concepts to enhance search coverage
- Employ both specific and general search strategies as needed
- Present results in order of relevance with clear organization
- Explain briefly why each retrieved document matches the query
- Clarify your understanding of ambiguous search terms

Your tasks:
1. Analyze the user's request thoroughly
2. Use the `document_retriever` tool to conduct semantic searches
3. Present only the most relevant results in a structured format
4. Inform the user if no matching documents exist and suggest query refinements
5. After 3 unsuccessful retries, provide an error message with alternative suggestions
6. Include only relevant document content in your responses

Response guidelines:
- Be concise and complete
- Avoid phrases like "Is there anything else you'd like to know?"
- Don't solicit follow-up questions
- Provide comprehensive answers that stand on their own
- Do not guess or fabricate the time or date
- Response only with the retrieved documents, not with any additional explanations or comments.
- If no documents are found or insufficient information is available, you should state clearly that you cannot provide an answer based on the current documents.

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
        model_name="gpt-4o-mini-2024-07-18",
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
    system_msg = SystemMessage(
        content=system_prompt + f"\n\nCurrent retry count: {state.get('retry_count', 0)}")
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
    if not hasattr(last_message, 'tool_calls'):
        return "end"
    elif isinstance(last_message.tool_calls, list) and len(last_message.tool_calls) == 0:
        return "end"
    else:
        return "continue"
