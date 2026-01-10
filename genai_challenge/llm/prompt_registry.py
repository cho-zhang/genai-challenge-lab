"""Prompt management for fixed and custom experimental prompts."""

from collections.abc import Callable

from genai_challenge.io.prompt_reader import PromptReader


class CustomPromptRegistry:
    """Registry for custom experimental prompts with name/version management."""

    def __init__(self):
        # Storage: {name: {version: prompt_string}}
        self._prompts: dict[str, dict[str, str]] = {}

    def register(self, name: str, version: str):
        """Decorator to register a custom prompt with a name and version.

        Args:
            name: The name identifier for the prompt
            version: The version identifier (e.g., "v1", "v2", "exp1")

        Usage:
            @custom_prompt_registry.register(name="test_generation", version="v1")
            def my_test_prompt():
                return "Generate tests for..."
        """

        def decorator(func: Callable[[], str]) -> Callable[[], str]:
            prompt_text = func()

            if not isinstance(prompt_text, str):
                raise TypeError(
                    f"Prompt must be a string, got {type(prompt_text)}"
                )

            if name not in self._prompts:
                self._prompts[name] = {}

            if version in self._prompts[name]:
                raise ValueError(
                    f"Custom prompt '{name}' version '{version}' already registered"
                )

            self._prompts[name][version] = prompt_text
            return func

        return decorator

    def get(self, name: str, version: str) -> str:
        """Fetch a custom prompt by name and version.

        Args:
            name: The name identifier for the prompt
            version: The version identifier

        Returns:
            The prompt string

        Raises:
            KeyError: If the prompt name or version is not found
        """
        if name not in self._prompts:
            raise KeyError(f"Custom prompt '{name}' not found in registry")

        if version not in self._prompts[name]:
            available_versions = list(self._prompts[name].keys())
            raise KeyError(
                f"Version '{version}' not found for custom prompt '{name}'. "
                f"Available versions: {available_versions}"
            )

        return self._prompts[name][version]


# Global registry instance for custom prompts
custom_prompt_registry = CustomPromptRegistry()


# ============================================================================
# Public function to get prompts
# ============================================================================


def get_custom_prompt(name: str, version: str) -> str:
    """Fetch a custom experimental prompt from the registry.

    This is for experimental prompts that you register and manage yourself
    using the @custom_prompt_registry.register decorator.

    Args:
        name: The name identifier for the custom prompt
        version: The version identifier

    Returns:
        The custom prompt string

    Raises:
        KeyError: If the prompt name or version is not found in registry
    """
    return custom_prompt_registry.get(name, version)


# ============================================================================
# Custom prompts
# ============================================================================


@custom_prompt_registry.register(name="custom_prompt", version="1")
def custom_prompt_1() -> str:
    return PromptReader.load_prompt(name="custom_prompt", version="1")


@custom_prompt_registry.register(name="custom_prompt", version="2")
def custom_prompt_2() -> str:
    return PromptReader.load_prompt(name="custom_prompt", version="2")


@custom_prompt_registry.register(name="custom_prompt", version="3")
def custom_prompt_3() -> str:
    return PromptReader.load_prompt(name="custom_prompt", version="3")


@custom_prompt_registry.register(name="custom_prompt", version="4")
def custom_prompt_4() -> str:
    return PromptReader.load_prompt(name="custom_prompt", version="4")


@custom_prompt_registry.register(name="custom_prompt", version="5")
def custom_prompt_5() -> str:
    return PromptReader.load_prompt(name="custom_prompt", version="5")
