import json
from unittest.mock import Mock, patch

import pytest
from litellm.utils import Choices, ModelResponse
from litellm.utils import Message as LiteLlmMessage

from genai_challenge.llm.model_clients import LLMClient
from genai_challenge.models.messages import (
    AssistantMessage,
    ImageUrlContent,
    LLMConfig,
    MessageRole,
    SystemMessage,
    TextContent,
    ToolCallInfo,
    ToolMessage,
    UserMessage,
)


class TestLLMClient:
    """Test suite for LLMClient class."""

    @pytest.fixture
    def llm_config(self):
        """Create a sample LLM configuration."""
        return LLMConfig(
            model_name="gpt-4",
            max_completion_tokens=100,
            temperature=0.7,
            top_p=0.9,
            tool_choice="auto",
        )

    @pytest.fixture
    def llm_client(self, llm_config):
        """Create an LLMClient instance."""
        return LLMClient(llm_config)

    def test_init(self, llm_config):
        """Test LLMClient initialization."""
        client = LLMClient(llm_config)
        assert client.config == llm_config
        assert client.config.model_name == "gpt-4"
        assert client.config.max_completion_tokens == 100

    def test_call_with_empty_messages_raises_error(self, llm_client):
        """Test that calling with empty messages raises ValueError."""
        with pytest.raises(ValueError, match="Messages cannot be empty"):
            llm_client.call([])

    @patch("genai_challenge.llm.model_clients.completion")
    def test_call_with_simple_user_message(self, mock_completion, llm_client):
        """Test calling LLM with a simple user message."""
        # Setup mock response
        mock_message = Mock(spec=LiteLlmMessage)
        mock_message.content = "Hello! How can I help you?"
        mock_message.tool_calls = None

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]
        mock_completion.return_value = mock_response

        # Call the method
        messages = [UserMessage(content="Hello")]
        result = llm_client.call(messages)

        # Assertions
        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello! How can I help you?"
        assert result.tool_calls is None
        mock_completion.assert_called_once()

    @patch("genai_challenge.llm.model_clients.completion")
    def test_call_with_system_and_user_messages(
        self, mock_completion, llm_client
    ):
        """Test calling LLM with system and user messages."""
        # Setup mock response
        mock_message = Mock(spec=LiteLlmMessage)
        mock_message.content = "I'm a helpful assistant."
        mock_message.tool_calls = None

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]
        mock_completion.return_value = mock_response

        # Call the method
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello"),
        ]
        result = llm_client.call(messages)

        # Assertions
        assert isinstance(result, AssistantMessage)
        assert result.content == "I'm a helpful assistant."

    def test_convert_input_system_message(self, llm_client):
        """Test converting system message to completion format."""
        messages = [SystemMessage(content="You are a helpful assistant.")]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == MessageRole.SYSTEM
        assert result[0]["content"] == "You are a helpful assistant."

    def test_convert_input_user_message_text_only(self, llm_client):
        """Test converting text-only user message."""
        messages = [UserMessage(content="Hello, world!")]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == MessageRole.USER
        assert result[0]["content"] == "Hello, world!"

    def test_convert_input_user_message_with_text_and_image(self, llm_client):
        """Test converting user message with text and image content."""
        messages = [
            UserMessage(
                content=[
                    TextContent(text="What's in this image?"),
                    ImageUrlContent(image_url="https://example.com/image.jpg"),
                ]
            )
        ]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == MessageRole.USER
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "What's in this image?"
        assert result[0]["content"][1]["type"] == "image_url"
        assert (
            result[0]["content"][1]["image_url"]["url"]
            == "https://example.com/image.jpg"
        )

    def test_convert_input_assistant_message_with_content(self, llm_client):
        """Test converting assistant message with content."""
        messages = [AssistantMessage(content="Hello! How can I help you?")]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == MessageRole.ASSISTANT
        assert result[0]["content"] == "Hello! How can I help you?"

    def test_convert_input_assistant_message_with_tool_calls(self, llm_client):
        """Test converting assistant message with tool calls."""
        tool_call = ToolCallInfo(
            id="call_123",
            tool_name="get_weather",
            tool_arguments={"location": "San Francisco"},
        )
        messages = [AssistantMessage(content=None, tool_calls=[tool_call])]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == MessageRole.ASSISTANT
        assert "content" not in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_123"
        assert result[0]["tool_calls"][0]["type"] == "function"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(
            result[0]["tool_calls"][0]["function"]["arguments"]
        ) == {"location": "San Francisco"}

    def test_convert_input_tool_message(self, llm_client):
        """Test converting tool message to completion format."""
        messages = [
            ToolMessage(
                tool_call_id="call_123",
                tool_name="get_weather",
                content='{"temperature": 72, "condition": "sunny"}',
            )
        ]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == MessageRole.TOOL
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["name"] == "get_weather"
        assert (
            result[0]["content"] == '{"temperature": 72, "condition": "sunny"}'
        )

    def test_convert_input_multiple_messages(self, llm_client):
        """Test converting multiple messages of different types."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What's the weather?"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCallInfo(
                        id="call_123",
                        tool_name="get_weather",
                        tool_arguments={"location": "San Francisco"},
                    )
                ],
            ),
            ToolMessage(
                tool_call_id="call_123",
                tool_name="get_weather",
                content='{"temperature": 72}',
            ),
        ]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 4
        assert result[0]["role"] == MessageRole.SYSTEM
        assert result[1]["role"] == MessageRole.USER
        assert result[2]["role"] == MessageRole.ASSISTANT
        assert result[3]["role"] == MessageRole.TOOL

    def test_convert_llm_response_with_text_content(self, llm_client):
        """Test converting LLM response with text content."""
        # Create mock response
        mock_message = Mock(spec=LiteLlmMessage)
        mock_message.content = "This is the response"
        mock_message.tool_calls = None

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]

        completion_params = {"model": "gpt-4", "messages": []}

        result = llm_client._convert_llm_response_to_assistant_message(
            model_response=mock_response, completion_params=completion_params
        )

        assert isinstance(result, AssistantMessage)
        assert result.content == "This is the response"
        assert result.tool_calls is None
        assert result.debug_llm_input == completion_params

    def test_convert_llm_response_with_tool_calls(self, llm_client):
        """Test converting LLM response with tool calls."""
        # Create mock tool call
        mock_function = Mock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "San Francisco"}'

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = mock_function

        mock_message = Mock(spec=LiteLlmMessage)
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]

        completion_params = {"model": "gpt-4", "messages": []}

        result = llm_client._convert_llm_response_to_assistant_message(
            model_response=mock_response, completion_params=completion_params
        )

        assert isinstance(result, AssistantMessage)
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].tool_name == "get_weather"
        assert result.tool_calls[0].tool_arguments == {
            "location": "San Francisco"
        }

    def test_convert_llm_response_with_dict_arguments(self, llm_client):
        """Test that passing dict arguments (instead of JSON string) should fail."""
        # Create mock tool call with dict arguments (invalid - should be string)
        mock_function = Mock()
        mock_function.name = "get_weather"
        mock_function.arguments = {"location": "Tokyo"}

        mock_tool_call = Mock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function = mock_function

        mock_message = Mock(spec=LiteLlmMessage)
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]

        completion_params = {"model": "gpt-4", "messages": []}

        # According to API docs, arguments should always be a JSON string, not a dict
        # This test expects the code to raise an error when dict is passed
        with pytest.raises(AssertionError):
            llm_client._convert_llm_response_to_assistant_message(
                model_response=mock_response,
                completion_params=completion_params,
            )

    def test_convert_llm_response_invalid_message_type(self, llm_client):
        """Test that invalid message type raises TypeError."""
        mock_choice = Mock(spec=Choices)
        mock_choice.message = "not a LiteLlmMessage"

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]

        completion_params = {"model": "gpt-4", "messages": []}

        with pytest.raises(TypeError, match="Expected LiteLLM message type"):
            llm_client._convert_llm_response_to_assistant_message(
                model_response=mock_response,
                completion_params=completion_params,
            )

    @patch("genai_challenge.llm.model_clients.completion")
    def test_call_passes_correct_parameters(self, mock_completion, llm_config):
        """Test that call method passes correct parameters to completion."""
        # Setup mock response
        mock_message = Mock(spec=LiteLlmMessage)
        mock_message.content = "Response"
        mock_message.tool_calls = None

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]
        mock_completion.return_value = mock_response

        # Create client with specific config
        client = LLMClient(llm_config)
        messages = [UserMessage(content="Test")]

        client.call(messages)

        # Check that completion was called with correct parameters
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_completion_tokens"] == 100
        assert call_args.kwargs["top_p"] == 0.9
        assert len(call_args.kwargs["messages"]) == 1
        assert "tool_choice" not in call_args.kwargs

    def test_convert_input_empty_user_content(self, llm_client):
        """Test converting user message with None content."""
        messages = [UserMessage(content=None)]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == MessageRole.USER
        assert result[0]["content"] == ""

    def test_convert_input_assistant_with_content_and_tool_calls(
        self, llm_client
    ):
        """Test converting assistant message with both content and tool calls."""
        tool_call = ToolCallInfo(
            id="call_789",
            tool_name="search",
            tool_arguments={"query": "python"},
        )
        messages = [
            AssistantMessage(
                content="Let me search for that.", tool_calls=[tool_call]
            )
        ]
        result = llm_client._convert_input_to_completion_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == MessageRole.ASSISTANT
        assert result[0]["content"] == "Let me search for that."
        assert len(result[0]["tool_calls"]) == 1

    def test_convert_llm_response_with_empty_content(self, llm_client):
        """Test converting LLM response with empty content."""
        mock_message = Mock(spec=LiteLlmMessage)
        mock_message.content = ""
        mock_message.tool_calls = None

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]

        completion_params = {"model": "gpt-4", "messages": []}

        result = llm_client._convert_llm_response_to_assistant_message(
            model_response=mock_response, completion_params=completion_params
        )

        assert result.content is None

    def test_convert_llm_response_with_multiple_tool_calls(self, llm_client):
        """Test converting LLM response with multiple tool calls."""
        # Create multiple mock tool calls
        mock_function1 = Mock()
        mock_function1.name = "get_weather"
        mock_function1.arguments = '{"location": "SF"}'

        mock_tool_call1 = Mock()
        mock_tool_call1.id = "call_1"
        mock_tool_call1.function = mock_function1

        mock_function2 = Mock()
        mock_function2.name = "get_weather"
        mock_function2.arguments = '{"location": "NYC"}'

        mock_tool_call2 = Mock()
        mock_tool_call2.id = "call_2"
        mock_tool_call2.function = mock_function2

        mock_message = Mock(spec=LiteLlmMessage)
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message

        mock_response = Mock(spec=ModelResponse)
        mock_response.choices = [mock_choice]

        completion_params = {"model": "gpt-4", "messages": []}

        result = llm_client._convert_llm_response_to_assistant_message(
            model_response=mock_response, completion_params=completion_params
        )

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[1].id == "call_2"
