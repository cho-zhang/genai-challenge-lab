"""Reader for experiment configuration YAML files."""
import yaml
from pathlib import Path
from typing import Union, Any

from genai_challenge.models.code_pilot_data import ExperimentConfig
from genai_challenge.models.messages import LLMConfig


class ExperimentConfigReader:
    """Reader for loading experiment configurations from YAML files."""

    @staticmethod
    def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
        """Load and validate experiment configuration from YAML file.

        Args:
            path: Path to the experiment_code_pilot.yaml file

        Returns:
            ExperimentConfig object with validated configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config is invalid or missing required fields
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML: expected a dictionary, got {type(raw)}")

        return ExperimentConfigReader._parse_experiment_config(raw)

    @staticmethod
    def _parse_experiment_config(raw: dict[str, Any]) -> ExperimentConfig:
        """Parse and validate raw YAML dict into ExperimentConfig.

        Args:
            raw: Raw dictionary from YAML file

        Returns:
            Validated ExperimentConfig object

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate llm_config section
        if "llm_config" not in raw:
            raise ValueError("Missing required field: 'llm_config'")

        if not isinstance(raw["llm_config"], dict):
            raise ValueError("'llm_config' must be a dictionary")

        llm_config_raw = raw["llm_config"]

        # model_name is required
        if "model_name" not in llm_config_raw:
            raise ValueError("Missing required field: 'llm_config.model_name'")

        # Build LLMConfig
        llm_config = LLMConfig(
            model_name=str(llm_config_raw["model_name"]),
            max_completion_tokens=llm_config_raw.get("max_completion_tokens"),
            reasoning_effort=llm_config_raw.get("reasoning_effort"),
            temperature=llm_config_raw.get("temperature"),
            top_p=llm_config_raw.get("top_p"),
            tool_choice=llm_config_raw.get("tool_choice"),
        )

        # Parse prompt configuration
        prompt_name = raw.get("prompt_name", "prompt_fixed")
        prompt_version = raw.get("prompt_version", "v1")

        if not isinstance(prompt_name, str):
            raise ValueError("'prompt_name' must be a string")

        if not isinstance(prompt_version, str):
            raise ValueError("'prompt_version' must be a string")

        # Parse file paths
        if "input_json_path" not in raw:
            raise ValueError("Missing required field: 'input_json_path'")

        if "output_folder" not in raw:
            raise ValueError("Missing required field: 'output_folder'")

        input_json_path = str(raw["input_json_path"])
        output_folder = str(raw["output_folder"])

        return ExperimentConfig(
            llm_config=llm_config,
            input_json_path=input_json_path,
            output_folder=output_folder,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )
