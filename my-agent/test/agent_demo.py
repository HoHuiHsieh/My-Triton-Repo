#!/usr/bin/env python3
"""
This is a demonstration script showing how to use the my_agent module.
"""
import sys
import os
sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), 
                        "/workspace/repository/my-agent-for-test/1/"
                        )
                    )
                )
import asyncio
import os
from typing import List
from my_agent import run_graph
from utils import set_variables


api_key = set_variables().get("API_KEY", "OPENAI_APIKEY")


# Example callback function that prints the content received
def print_callback(content: str):
    print("----------------------------------------")
    print(content)

# Sample conversation messages
messages = [
    # {"role": "user", "content": "Find a movie about psychologist in document, how old is the movie?"},
    # {"role": "user", "content": "9 + 8 - 7 * 6 / 5 + 4 * 3 - 2 * 1 = ?"},
    {"role": "user", "content": "How many '.' in s...t.ra..wb.e..r.r.y.?"},
]

async def main():
    # Get the OpenAI API key
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it by running: export OPENAI_API_KEY='your-api-key'")
        return
    
    print("Starting the agent workflow...")
    
    # Run the graph with the input messages, API key, and callback
    await run_graph(
        input=messages,
        openai_apikey=api_key,
        callback=print_callback
    )
    
    print("Agent workflow completed.")

if __name__ == "__main__":
    asyncio.run(main())
