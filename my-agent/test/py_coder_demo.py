#!/usr/bin/env python3
"""
This is a demonstration script showing how to use the py_coder module.
"""
import sys
import os
sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), 
                        "/workspace/repository/my-agent-for-test/1/my_agent/"
                        )
                    )
                )
from py_coder import graph
from py_coder.state import PyCoderState
from langchain_core.messages import HumanMessage, AIMessage
from utils import set_variables


def run_pycoder_demo():
    """
    Demonstrates how to use the py_coder module by running a simple example.
    """
    # You should set your API key here or in the environment
    api_key = set_variables().get("API_KEY", "OPENAI_APIKEY")
    
    # Create an initial state with an assistant message
    state = PyCoderState(
        messages=[
            HumanMessage(content="remember x = 10"),
            AIMessage(content="ok, I will compute the sum of numbers from 1 to x."),
        ],
        api_key=api_key
    )
    
    print("Initial state:", state)
    print("\nRunning the py_coder graph...\n")
    
    # Execute the graph
    try:
        final_state = graph.invoke(state)
        print("\nFinal state:")
        for msg in final_state["messages"]:
            print(f"\n{msg.type}: {msg.content}")

        print("\nDone! Check the final state for results.")
    except Exception as e:
        print(f"Error running the graph: {str(e)}")

if __name__ == "__main__":
    print("PyCoder Module Demo")
    print("===================")
    run_pycoder_demo()
