from pathlib import Path


class PromptReader:
    """Reader for loading prompt text files with versioning support."""

    # Default directory for prompt files (relative to this module)
    DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"

    @staticmethod
    def load_prompt(
        name: str, version: str, prompts_dir: str | Path | None = None
    ) -> str:
        """Load a prompt text file by name and version.

        Looks for files matching pattern: <name>_<version>.txt

        Args:
            name: The prompt name (e.g., "test_generation")
            version: The prompt version (e.g., "v1", "1.0")
            prompts_dir: Optional custom directory for prompts. If not provided,
                        uses genai_challenge/data/prompts/

        Returns:
            The full text content of the prompt file

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            ValueError: If name or version is empty
            IOError: If the file cannot be read

        Example:
            >>> prompt = PromptReader.load_prompt("test_generation", "v1")
            >>> print(prompt)
        """
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Prompt name cannot be empty")
        if not version or not version.strip():
            raise ValueError("Prompt version cannot be empty")

        # Determine prompts directory
        if prompts_dir is None:
            prompts_dir = PromptReader.DEFAULT_PROMPTS_DIR
        prompts_dir = Path(prompts_dir)

        # Check if prompts directory exists
        if not prompts_dir.exists():
            raise FileNotFoundError(
                f"Prompts directory not found: {prompts_dir}. "
                f"Please create the directory or provide a valid prompts_dir path."
            )

        # Construct file path
        filename = f"{name}_{version}.txt"
        file_path = prompts_dir / filename

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {file_path}\n"
                f"Expected pattern: {name}_<version>.txt in {prompts_dir}"
            )

        # Read and return file content
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            raise OSError(f"Failed to read prompt file {file_path}: {e}") from e
