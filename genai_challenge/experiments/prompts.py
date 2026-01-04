from genai_challenge.llm.prompt_registry import custom_prompt_registry


@custom_prompt_registry.register(name="test_generation", version="v1")
def test_generation_v1():
    return """Generate pytest tests for the given function specification.

        Requirements:
        - Use pytest framework
        - Test normal cases, edge cases, and error conditions
        - Include the testing import statement exactly once
        - Do not implement the target function
        - Return clean Python code without markdown fences
        
        Format your test code between these delimiters:
        ###|=-=-=beginning of tests=-=-=|
        <your test code here>
        ###|=-=-=end of tests=-=-=|
    """