import json
from pathlib import Path

from genai_challenge.models.code_pilot_data import InputCodeList, InputData


class NISTInputReader:
    @staticmethod
    def load_code_pilot_input(path: str | Path) -> InputData:
        """
        Load and validate NIST GenAI Code Pilot input JSON file.
        """
        path = Path(path)
        file_name = path.name
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Top-level {file_name} is not valid JSON.")

        return NISTInputReader._validate_input_data(raw, file_name)

    @staticmethod
    def _validate_input_data(raw: dict, file_name: str) -> InputData:
        required_fields = ["name", "version", "Evaluation_Version", "code_list"]
        for field in required_fields:
            if field not in raw:
                raise ValueError(
                    f"Missing required field: '{field}' in {file_name}"
                )

        if not isinstance(raw["code_list"], list):
            raise ValueError("'code_list' must be a list")

        trials = []
        for idx, item in enumerate(raw["code_list"]):
            try:
                trials.append(NISTInputReader._parse_trial(item, file_name))
            except Exception as e:
                raise ValueError(
                    f"Failed to parse trial at index {idx}: {e}"
                ) from e

        return InputData(
            name=str(raw["name"]),
            version=str(raw["version"]),
            Evaluation_Version=str(raw["Evaluation_Version"]),
            code_list=trials,
        )

    @staticmethod
    def _parse_trial(item: dict, file_name: str) -> InputCodeList:
        if not isinstance(item, dict):
            raise TypeError(f"Trial is not a valid JSON in {file_name}")

        required_fields = [
            "trial_id",
            "testing_import_statement",
            "primary_method_name",
            "specification",
            "prompt_fixed",
        ]
        for field in required_fields:
            if field not in item:
                raise ValueError(
                    f"Missing required field: '{field}' in {file_name}"
                )

        return InputCodeList(
            trial_id=str(item["trial_id"]),
            testing_import_statement=item["testing_import_statement"].strip(),
            primary_method_name=item["primary_method_name"].strip(),
            specification=item["specification"].strip(),
            prompt_fixed=item["prompt_fixed"].strip(),
        )
