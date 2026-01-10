from dataclasses import dataclass

from genai_challenge.models.messages import LLMConfig


@dataclass
class InputCodeList:
    trial_id: str
    primary_method_name: str
    testing_import_statement: str
    specification: str
    prompt_fixed: str


@dataclass
class InputData:
    name: str
    Evaluation_Version: str
    code_list: list[InputCodeList]
    version: str = "2.00"


@dataclass
class SubmissionCodeList:
    trial_id: str
    prompt_number: str
    prompt: str
    primary_method_name: str
    test_output: str
    test_code: str


@dataclass
class SubmissionData:
    name: str
    system: str
    code_list: list[SubmissionCodeList]
    version: str = "2.00"


@dataclass
class ExperimentConfig:
    """Configuration for NIST code pilot experiments.

    Loaded from experiment_code_pilot.yaml to control:
    - Which LLM model and parameters to use (via llm_config)
    - Which prompt to use (fixed from NIST data or custom from registry)
    - Input and output file paths
    """

    llm_config: LLMConfig
    input_json_path: str
    output_folder: str
    experiment_version: str = ""  # Auto-generated at runtime if not provided
    prompt_name: str = (
        "prompt_fixed"  # default to "prompt_fixed" for NIST fixed prompts
    )
    prompt_version: str = "v1"  # version identifier for custom prompts
