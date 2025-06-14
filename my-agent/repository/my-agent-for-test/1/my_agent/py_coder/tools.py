from langchain_core.tools import tool
from pydantic import BaseModel, Field


class PyCoderInputSchema(BaseModel):
    code: str = Field(
        description="python code to execute, must define a variable named 'result' to store the output.")


def get_pycoder_tool() -> str:
    """Get a Python coder tool function."""

    @tool("python_coder", args_schema=PyCoderInputSchema, response_format="content")
    def python_coder(code: str) -> str:
        """
        Call this tool to execute Python code and return the result.
        """
        try:
            local_vars = {}
            exec(code, {}, local_vars)
            result = local_vars.get('result', None)
            if result is None:
                raise Exception("Error: Failed to execute code or no result variable defined.")
            return f"\n```python\n{code}\n```\nPython execution result: {result}\n"
        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            return f"\n```python\n{code}\n```\nPython execution error: {error_type}: {error_message}\n"

    return python_coder
