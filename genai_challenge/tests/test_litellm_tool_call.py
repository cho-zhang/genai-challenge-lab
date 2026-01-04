import litellm
import json

from genai_challenge.llm.model_clients import LLMClient
from genai_challenge.models.messages import LLMConfig, SystemMessage



def test_parallel_function_call_disabled():
    # https://docs.litellm.ai/docs/completion/function_call#full-code---parallel-function-calling-with-gpt-35-turbo-1106
    # Example dummy function hard coded to return the same weather
    # In production, this could be your backend API or an external API
    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
        elif "san francisco" in location.lower():
            return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
        elif "paris" in location.lower():
            return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

    try:
        # Step 1: send the conversation and available functions to the model
        messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        response = litellm.completion(
            model="gpt-5-nano",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
            parallel_tool_calls=False,
        )
        print("\nFirst LLM Response:\n", response)
        # First LLM Response:
        #    ModelResponse(id='chatcmpl-Cts....itUZtvLaNry', created=1767433407, model='gpt-5-nano-2025-08-07', object='chat.completion', choices=[Choices(finish_reason='tool_calls', index=0, message=Message(content=None, role='assistant', tool_calls=[ChatCompletionMessageToolCall(function=Function(arguments='{"location":"San Francisco, CA"}', name='get_current_weather'), id='call_Tu0Prd...9G', type='function')], ...), ...], usage=Usage(...), service_tier='default')
        # Length of tool calls 1
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        print("\nLength of tool calls", len(tool_calls))

        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "get_current_weather": get_current_weather,
            }  # only one function in this example, but you can have multiple
            messages.append(response_message)  # extend conversation with assistant's reply

            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            second_response = litellm.completion(
                model="gpt-5-nano-2025-08-07",
                messages=messages,
            )  # get a new response from the model where it can see the function response
            print("\nSecond LLM response:\n", second_response)
            # Second LLM response:
            #  ModelResponse(id='chatcmpl-...', created=1767428031, model='gpt-5-nano-2025-08-07', object='chat.completion', system_fingerprint='fp_982035f36f', choices=[Choices(finish_reason='stop', index=0, message=Message(content='Currently, the weather in San Francisco is 72°F (22°C), in Tokyo is 10°C, and in Paris is 22°C.', role='assistant', tool_calls=None, function_call=None, provider_specific_fields={'refusal': None}, annotations=[]), provider_specific_fields={})], usage=Usage(completion_tokens=30, prompt_tokens=175, total_tokens=205, completion_tokens_details=CompletionTokensDetailsWrapper(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0, text_tokens=None, image_tokens=None), prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=0, cached_tokens=0, text_tokens=None, image_tokens=None)), service_tier='default')
            return second_response
    except Exception as e:
      print(f"Error occurred: {e}")


def test_generate_text_with_fixed_prompt():
    prompt = (
        "We have python code that implements the following specification.\n\nSpecification:\n\ndef add(x: int, "
        "y: int) -> int:\n    \"\"\"\n    Given two integers x, and y, return the sum of x and y. If either x "
        "or y is not\n    an integer, raise a TypeError Exception.\n    \"\"\"\n\n\nPlease write python pytest "
        "test code that comprehensively tests the code for method add to determine if the code satisfies the "
        "specification or not. When writing tests:\n* write a comprehensive test suite,\n* test edge cases,"
        "\n* only generate correct tests, and\n* include tests for TypeError cases.\n\nPlease write "
        "'###|=-=-=beginning of tests=-=-=|' before the tests. Write '###|=-=-=end of tests=-=-=|' immediately "
        "after the tests. Import any needed packages, including pytest. Import the code being tested by adding "
        "the line `from genai_code_file import add` the line after '###|=-=-=beginning of tests=-=-=|'. Do not "
        "provide an implementation of the method add with the tests."
    )
    llm_config = LLMConfig(
        model_name="gpt-5-nano",
    )
    llm_client = LLMClient(llm_config)

    assistant_message = llm_client.call(
        messages=[
            SystemMessage(
                role="system",
                content=prompt,
            )
        ]
    )
    print(f"\nAssistant message for fixed prompt:\n{assistant_message}")

test_parallel_function_call_disabled()
test_generate_text_with_fixed_prompt()
