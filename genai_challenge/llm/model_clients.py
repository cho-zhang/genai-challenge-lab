import json
import logging
from typing import Any

import litellm
from litellm import completion
from litellm.utils import Message as LiteLlmMessage
from litellm.utils import ModelResponse
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    wait_random_exponential,
)

from genai_challenge.models.messages import (
    AssistantMessage,
    ImageUrlContent,
    LLMConfig,
    MessageRole,
    RoleMessage,
    TextContent,
    ToolCallInfo,
    ToolMessage,
)
from pipelines.retry_utils import is_retryable_exception

logger = logging.getLogger(__name__)


def _log_retry_attempt(retry_state: RetryCallState):
    ex = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"Encountering transient LLM API error: {ex},"
        f"retrying... (attempt {retry_state.attempt_number})"
    )


class LLMClient:
    """Client for interacting with LLM models via LiteLLM."""

    def __init__(self, config: LLMConfig):
        self.config = config

        # Configure litellm to drop unsupported parameters for different models
        # litellm.exceptions.UnsupportedParamsError: litellm.UnsupportedParamsError: gpt-5 models
        # don't support temperature=0.7. Only temperature=1 is supported. For gpt-5.1, temperature is supported when
        # reasoning_effort='none' (or not specified, as it defaults to 'none').
        # To drop unsupported params set litellm.drop_params = True
        litellm.drop_params = True

    @retry(
        retry=retry_if_exception(is_retryable_exception),
        reraise=True,
        stop=stop_after_attempt(5),  # Allow more retries for rate limits
        wait=wait_random_exponential(
            multiplier=1, min=4, max=60
        ),  # Wait 4-60 seconds with exponential backoff
        before_sleep=_log_retry_attempt,
    )
    def call(
        self,
        messages: list[RoleMessage],
        available_tools: list | None = None,
    ) -> AssistantMessage:
        """Call LLM with a list of messages and return an AssistantMessage."""
        if not messages:
            raise ValueError("Messages cannot be empty")

        completion_messages = self._convert_input_to_completion_messages(
            messages
        )

        # Build completion parameters
        completion_params = {
            "model": self.config.model_name,
            "messages": completion_messages,
            "max_completion_tokens": self.config.max_completion_tokens,
            "reasoning_effort": self.config.reasoning_effort,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if available_tools:
            # litellm.llms.openai.common_utils.OpenAIError: Error code: 400 -
            # {'error': {'message': "Invalid value for 'tool_choice': 'tool_choice' is only allowed when
            # 'tools' are specified.", 'type': 'invalid_request_error', 'param': 'tool_choice', 'code': None}}
            completion_params["tool_choice"] = self.config.tool_choice
            completion_params["tools"] = available_tools

        response = completion(**completion_params)

        assistant_message = self._convert_llm_response_to_assistant_message(
            completion_params=completion_params,
            model_response=response,
        )
        return assistant_message

    def _convert_input_to_completion_messages(
        self, messages: list[RoleMessage]
    ) -> list[dict[str, Any]]:
        """Convert typed RoleMessage list to dict format for litellm completion API"""
        result = []

        for msg in messages:
            message_dict: dict[str, Any] = {"role": msg.role}

            if msg.role == MessageRole.SYSTEM or msg.role == MessageRole.USER:
                if isinstance(msg.content, list):
                    # content is list[MESSAGE_CONTENT]
                    content_list = []
                    for content_item in msg.content:
                        if isinstance(content_item, TextContent):
                            content_list.append(
                                {"type": "text", "text": content_item.text}
                            )
                        elif isinstance(content_item, ImageUrlContent):
                            content_list.append(
                                {
                                    "type": "image_url",
                                    # bug: "image_url": content_item.image_url
                                    "image_url": {
                                        "url": content_item.image_url
                                    },
                                }
                            )
                    message_dict["content"] = content_list
                else:
                    # text-only user|system message
                    message_dict["content"] = msg.content or ""

            elif isinstance(msg, AssistantMessage):
                # Handle assistant messages with optional content and tool_calls
                if msg.content is not None:
                    message_dict["content"] = msg.content

                if msg.tool_calls:
                    tool_calls_list = []
                    for tc in msg.tool_calls:
                        # https://docs.litellm.ai/docs/completion/function_call
                        tool_calls_list.append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.tool_name,
                                    # "arguments": tc.tool_arguments
                                    "arguments": json.dumps(tc.tool_arguments),
                                },
                            }
                        )
                    message_dict["tool_calls"] = tool_calls_list

            elif isinstance(msg, ToolMessage):
                # Handle tool response messages
                message_dict["tool_call_id"] = msg.tool_call_id
                message_dict["name"] = msg.tool_name
                message_dict["content"] = msg.content

            result.append(message_dict)

        return result

    def _convert_llm_response_to_assistant_message(
        self, model_response: ModelResponse, completion_params: dict[str, Any]
    ) -> AssistantMessage:
        """Convert litellm response to AssistantMessage.

        Args:
            model_response: The response object from litellm.completion()
            completion_params: The parameters sent to the LLM completion call

        Returns:
            AssistantMessage with content and/or tool_calls extracted from response
        """
        message = model_response.choices[0].message
        debug_llm_output = (
            model_response
            if isinstance(model_response, dict)
            else {"response": model_response}
        )
        debug_llm_input = completion_params

        if not isinstance(message, LiteLlmMessage):
            raise TypeError(f"Expected LiteLLM message type. Actual: {message}")

        # Extract content
        content = message.content if message.content else None

        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                # Parse function arguments (litellm returns them as JSON string)
                assert isinstance(tool_call.function.arguments, str)
                function_args = json.loads(tool_call.function.arguments)
                tool_calls.append(
                    ToolCallInfo(
                        id=tool_call.id,
                        tool_name=tool_call.function.name,
                        tool_arguments=function_args,
                    )
                )

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            debug_llm_input=debug_llm_input,
            debug_llm_output=debug_llm_output,
        )
