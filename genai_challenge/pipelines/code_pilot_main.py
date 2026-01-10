import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Union

from genai_challenge.io.nist_reader import NISTInputReader
from genai_challenge.io.experiment_config_reader import ExperimentConfigReader
from genai_challenge.llm.model_clients import LLMClient
from genai_challenge.llm.prompt_registry import get_custom_prompt
from genai_challenge.models.code_pilot_data import (
    InputCodeList,
    SubmissionData,
    SubmissionCodeList,
    ExperimentConfig,
)
from genai_challenge.models.messages import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    RoleMessage,
)

TESTS_CODE_BEGIN = r"###\|\=-=-=beginning of tests=-=-=\|"
TESTS_CODE_END = r"###\|\=-=-=end of tests=-=-=\|"


def _extract_test_code(text: str) -> str:
    """Extract test code from LLM output using delimiters.

    If both delimiters exist, extract code between them.
    If only TESTS_CODE_BEGIN exists, extract everything after it.
    If no delimiters exist, raise an error.

    Args:
        text: LLM response text

    Returns:
        Extracted test code

    Raises:
        ValueError: If TESTS_CODE_BEGIN delimiter is not found
    """
    # First try to match both delimiters
    match = re.search(
        rf"{TESTS_CODE_BEGIN}(.*?){TESTS_CODE_END}",
        text,
        flags=re.DOTALL
    )
    if match:
        return match.group(1).strip()

    # If no end delimiter, try to match just the beginning and take everything after
    match_begin = re.search(
        rf"{TESTS_CODE_BEGIN}(.*)",
        text,
        flags=re.DOTALL
    )
    if match_begin:
        return match_begin.group(1).strip()

    # No beginning delimiter found - this is an error
    raise ValueError("Could not find test code beginning delimiter in LLM output")


def _validate_test_code(
    raw_code: str,
    testing_import_statement: str,
    primary_method_name: str,
) -> str | None:
    """Validate test code based on eval requirements.

    - pytest is imported
    - testing_import_statement is included exactly once
    - No implementation of target function exists
    - Code is clean text (no markdown fences)
    - Test function for target function exists
    - Test code is restricted to 25,000 characters or less

    Args:
        raw_code: The extracted test code to validate
        testing_import_statement: The expected import statement for the target function

    Returns:
        None if validation passes, error message string if validation fails
    """
    errors = []

    # Check 1: pytest is imported
    if "import pytest" not in raw_code:
        errors.append("Missing pytest import")

    # Check 2: testing_import_statement is included exactly once
    # TODO: LLM may optimize by removing unused imports or adding spaces
    # import_count = raw_code.count(testing_import_statement)
    # if import_count == 0:
    #     errors.append(f"Missing required import: {testing_import_statement}")
    # elif import_count > 1:
    #     errors.append(f"Import statement appears {import_count} times (should be exactly once): {testing_import_statement}")

    # Check 3: No implementation of primary_method_name exists
    if re.search(rf'^def\s+{primary_method_name}\s*\(', raw_code, re.MULTILINE):
        errors.append(f"Test code should not contain implementation of function: {primary_method_name}")

    # Check 4: Code is clean text (no markdown fences)
    if "```" in raw_code:
        errors.append("Test code contains markdown code fences (```)")

    # Check 5: Test functions exist
    if not re.search(r'^\s*def\s+test_\w+\s*\(', raw_code, re.MULTILINE):
        errors.append(f"No test functions found for target method: {primary_method_name}")

    # Check 6: Restricted to 25,000 characters or less
    if len(raw_code) > 25000:
        errors.append(f"test code size is > 25,000 characters.")

    # Return error message if any checks failed
    if errors:
        error_message = "Validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        print(error_message)
        return f"\n{error_message}\n\n{raw_code})"

    return None


def _build_completion_messages(
    input_code: InputCodeList,
    config: ExperimentConfig
) -> list[RoleMessage]:
    """Build messages for LLM call based on prompt configuration.

    For fixed prompts:
        - No system message
        - User message contains complete prompt from NIST data

    For custom prompts (registry):
        - System message contains instructions from registry
        - User message contains testing_import_statement + specification

    Args:
        input_code: Input problem specification
        config: Experiment configuration with prompt settings

    Returns:
        List of messages ready for LLM call
    """
    if config.prompt_name == "prompt_fixed":
        # NIST fixed prompt - complete prompt already provided
        return [UserMessage(content=input_code.prompt_fixed)]

    # Custom prompt: fetch system message from registry + build user message
    system_message = SystemMessage(
        content=get_custom_prompt(
            name=config.prompt_name,
            version=config.prompt_version
        )
    )

    user_message = UserMessage(
        content=f"""Testing Import Statement:
{input_code.testing_import_statement}
            
Specification:
{input_code.specification}
        """
    )

    return [system_message, user_message]


def _build_prompt_string(messages: list[RoleMessage]) -> str:
    """Build a single string representation of the prompt from messages.

    For fixed prompts (single user message), returns the user message content.
    For custom prompts (system + user messages), concatenates them.

    Args:
        messages: List of messages used in LLM call

    Returns:
        Single string representation of the prompt
    """
    if len(messages) == 1:
        # Fixed prompt - just the user message
        return messages[0].content
    else:
        # Custom prompt - concatenate system and user messages
        return "\n\n".join(msg.content for msg in messages)


def _convert_to_submission_code(
    input_code_list: InputCodeList,
    assistant_message: AssistantMessage,
    prompt_number: str,
    prompt_used: str
) -> SubmissionCodeList:
    """Convert LLM response to SubmissionCodeList for submission.

    Args:
        input_code_list: Original input problem specification
        assistant_message: LLM response message
        prompt_number: Prompt number as string ("0" for fixed, "1"+ for custom)
        prompt_used: The actual prompt that was used for this LLM call

    Returns:
        SubmissionCodeList ready for submission

    Raises:
        ValueError: If test code cannot be extracted from response
    """
    if not assistant_message.content:
        raise ValueError(f"No content in assistant message for trial {input_code_list.trial_id}")

    test_code = _extract_test_code(assistant_message.content)
    error_message = _validate_test_code(
        raw_code=test_code,
        testing_import_statement=input_code_list.testing_import_statement,
        primary_method_name=input_code_list.primary_method_name,
    )

    return SubmissionCodeList(
        trial_id=input_code_list.trial_id,
        prompt_number=prompt_number,
        prompt=prompt_used,
        primary_method_name=input_code_list.primary_method_name,
        test_output=assistant_message.content,
        test_code=test_code if not error_message else ""
    )


def _get_project_root() -> Path:
    """Get the project root directory.

    Uses environment variable PROJECT_ROOT if set, otherwise finds the
    parent directory of the genai_challenge package. This works regardless
    of where the script is run from and in cloud environments.

    Returns:
        Path to the project root directory
    """
    # Option 1: Use environment variable if set
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"])

    # Option 2: Find via genai_challenge package location
    import genai_challenge
    package_path = Path(genai_challenge.__file__).parent
    return package_path.parent


def _resolve_paths_from_config(config: ExperimentConfig) -> tuple[Path, Path]:
    """Resolve input and output paths from experiment config.

    If paths are absolute, use them as-is. If relative, resolve from project root.

    Args:
        config: Experiment configuration with input_json_path and output_folder

    Returns:
        Tuple of (input_json_path, output_folder)
    """
    project_root = _get_project_root()
    print(f"  ✓ Project root: {project_root}")

    # Resolve input JSON path
    input_json_path = Path(config.input_json_path)
    if not input_json_path.is_absolute():
        input_json_path = project_root / input_json_path

    # Resolve output folder
    output_folder = Path(config.output_folder)
    if not output_folder.is_absolute():
        output_folder = project_root / output_folder

    return input_json_path, output_folder


def save_submission_data(
    submission_data: SubmissionData,
    output_folder: Union[str, Path]
) -> None:
    """Save SubmissionData to JSON file.

    Args:
        submission_data: The submission data to save
        output_folder: Folder where JSON file should be written (filename will be submission_data.name)
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Use submission_data.name as filename
    output_path = output_folder / f"{submission_data.name}.json"

    # Convert dataclass to dict for JSON serialization
    submission_dict = {
        "name": submission_data.name,
        "version": submission_data.version,
        "system": submission_data.system,
        "code_list": [
            {
                "trial_id": code.trial_id,
                "prompt_number": code.prompt_number,
                "prompt": code.prompt,
                "primary_method_name": code.primary_method_name,
                "test_output": code.test_output,
                "test_code": code.test_code,
            }
            for code in submission_data.code_list
        ]
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(submission_dict, f, indent=2, ensure_ascii=False)

    print(f"Submission saved to: {output_path}")


def run_code_pilot(
    experiment_yaml_path: Union[str, Path]
) -> SubmissionData:
    """Main orchestration function for NIST Code Pilot challenge.

    Workflow:
    1. Read experiment yaml file to fetch configs (including input/output paths)
    2. Read input JSON and load InputData
    3. Loop each InputCodeList from InputData
    4. Construct messages from prompt (fixed or custom)
    5. Call LLM with messages
    6. Convert AssistantMessage to SubmissionCodeList
    7. Persist SubmissionData to JSON

    Args:
        experiment_yaml_path: Path to experiment config YAML file

    Returns:
        SubmissionData object with all generated tests
    """
    print("=" * 80)
    print("NIST GenAI Code Pilot Challenge - Generation Pipeline")
    print("=" * 80)

    # Step 1: Load experiment configuration
    print(f"\n[1/5] Loading experiment config from: {experiment_yaml_path}")
    config = ExperimentConfigReader.load_experiment_config(experiment_yaml_path)

    # Generate experiment version timestamp
    config.experiment_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get input/output paths from config
    input_json_path, output_folder = _resolve_paths_from_config(config)

    print(f"  ✓ Model: {config.llm_config.model_name}")
    print(f"  ✓ Temperature: {config.llm_config.temperature}")
    print(f"  ✓ Prompt: {config.prompt_name} (v{config.prompt_version})")
    print(f"  ✓ Experiment version: {config.experiment_version}")
    print(f"  ✓ Input: {input_json_path}")
    print(f"  ✓ Output folder: {output_folder}")

    # Step 2: Load input data
    print("\n[2/5] Loading input data...")
    input_data = NISTInputReader.load_code_pilot_input(input_json_path)
    print(f"  ✓ Loaded {len(input_data.code_list)} problems")

    # Initialize LLM client
    llm_client = LLMClient(config.llm_config)

    # Step 3-6: Process each problem
    print(f"\n[3/5] Processing {len(input_data.code_list)} problems...")
    submission_code_list = []

    for idx, input_code in enumerate(input_data.code_list, start=1):
        print(f"\n  [{idx}/{len(input_data.code_list)}] Processing trial: {input_code.trial_id}")
        print(f"      Primary method name: {input_code.primary_method_name}")

        try:
            # Call LLM twice: once with fixed prompt, once with custom prompt
            # === FIXED PROMPT (prompt_number = "0") ===
            print("      Calling LLM with fixed prompt...")

            # if input_code.trial_id != "00003_make_palindrome":
            #     continue

            # Build messages for fixed prompt (just a single UserMessage)
            fixed_messages = [UserMessage(content=input_code.prompt_fixed)]
            fixed_prompt_string = input_code.prompt_fixed

            # Call LLM with fixed prompt
            fixed_assistant_message = llm_client.call(fixed_messages)
            print(f"      ✓ Received response ({len(fixed_assistant_message.content or '')} chars)")

            # Convert to submission format with prompt_number "0"
            fixed_submission_code = _convert_to_submission_code(
                input_code_list=input_code,
                assistant_message=fixed_assistant_message,
                prompt_number="0",
                prompt_used=fixed_prompt_string
            )
            submission_code_list.append(fixed_submission_code)
            print(f"      ✓ Extracted test code ({len(fixed_submission_code.test_code)} chars)")

            # === CUSTOM PROMPT (prompt_number = "1") ===
            print("      Calling LLM with custom prompt...")

            # Build messages for custom prompt
            custom_messages = _build_completion_messages(input_code, config)
            custom_prompt_string = _build_prompt_string(custom_messages)

            # Call LLM with custom prompt
            custom_assistant_message = llm_client.call(custom_messages)
            print(f"      ✓ Received response ({len(custom_assistant_message.content or '')} chars)")

            # Convert to submission format with prompt_number "1"
            custom_submission_code = _convert_to_submission_code(
                input_code_list=input_code,
                assistant_message=custom_assistant_message,
                prompt_number="1",
                prompt_used=custom_prompt_string
            )
            submission_code_list.append(custom_submission_code)
            print(f"      ✓ Extracted test code ({len(custom_submission_code.test_code)} chars)")

            # break
            # time.sleep(25)  # TODO: sleep based on Usage tokens

        except Exception as e:
            print(f"      ✗ Error processing trial {input_code.trial_id}: {e}")
            raise

    # Step 7: Create submission data
    print("\n[4/5] Creating submission data...")
    model_name = config.llm_config.model_name
    reasoning = config.llm_config.reasoning_effort
    prompt_name = config.prompt_name
    prompt_vers = config.prompt_version
    submission_name = f"submission_{config.experiment_version}_{model_name}_reasoning-{reasoning}_{prompt_name}_{prompt_vers}"
    submission_data = SubmissionData(
        name=submission_name,
        version="2.00",
        system="TICCZ-CHO-Z-nist-code-pilot-pipeline",
        code_list=submission_code_list
    )
    print(f"  ✓ Created submission with {len(submission_data.code_list)} test suites")

    # Save to JSON
    output_file_path = output_folder / f"{submission_data.name}.json"
    print(f"\n[5/5] Saving submission to: {output_file_path}")
    save_submission_data(submission_data, output_folder)
    print("  ✓ Submission saved successfully")

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)

    return submission_data


if __name__ == "__main__":
    # Default path to experiment config (contains input/output paths)
    EXPERIMENT_YAML = Path(__file__).parent.parent / "experiment_code_pilot.yaml"

    # Run pipeline (all paths are configured in the YAML file)
    run_code_pilot(experiment_yaml_path=EXPERIMENT_YAML)
