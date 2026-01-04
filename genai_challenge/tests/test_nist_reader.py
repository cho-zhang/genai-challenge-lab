"""Unit tests for NISTInputReader."""

import json
import pytest

from genai_challenge.io.nist_reader import NISTInputReader
from genai_challenge.models.code_pilot_data import InputData, InputCodeList


class TestNISTInputReader:
    """Test suite for NISTInputReader class."""

    @pytest.fixture
    def valid_input_data(self):
        """Valid input data fixture."""
        return {
            "name": "Test Input",
            "version": "2.00",
            "Evaluation_Version": "Pilot1",
            "code_list": [
                {
                    "trial_id": "00001_add",
                    "testing_import_statement": "from genai_code_file import add",
                    "primary_method_name": "add",
                    "specification": "def add(x: int, y: int) -> int:\n    pass",
                    "prompt_fixed": "Write tests for add function",
                }
            ],
        }

    @pytest.fixture
    def valid_trial_data(self):
        """Valid trial data fixture."""
        return {
            "trial_id": "00001_add",
            "testing_import_statement": "from genai_code_file import add",
            "primary_method_name": "add",
            "specification": "def add(x: int, y: int) -> int:\n    pass",
            "prompt_fixed": "Write tests for add function",
        }

    def test_load_valid_input(self, valid_input_data, tmp_path):
        """Test loading valid input JSON file."""
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        result = NISTInputReader.load_code_pilot_input(input_file)

        assert isinstance(result, InputData)
        assert result.name == "Test Input"
        assert result.version == "2.00"
        assert result.Evaluation_Version == "Pilot1"
        assert len(result.code_list) == 1
        assert isinstance(result.code_list[0], InputCodeList)
        assert result.code_list[0].trial_id == "00001_add"

    def test_load_with_string_path(self, valid_input_data, tmp_path):
        """Test loading with string path instead of Path object."""
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        result = NISTInputReader.load_code_pilot_input(str(input_file))

        assert isinstance(result, InputData)
        assert result.name == "Test Input"

    def test_load_multiple_trials(self, valid_input_data, tmp_path):
        """Test loading input with multiple trials."""
        valid_input_data["code_list"].append(
            {
                "trial_id": "00002_multiply",
                "testing_import_statement": "from genai_code_file import multiply",
                "primary_method_name": "multiply",
                "specification": "def multiply(x: int, y: int) -> int:\n    pass",
                "prompt_fixed": "Write tests for multiply function",
            }
        )
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        result = NISTInputReader.load_code_pilot_input(input_file)

        assert len(result.code_list) == 2
        assert result.code_list[0].trial_id == "00001_add"
        assert result.code_list[1].trial_id == "00002_multiply"

    def test_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            NISTInputReader.load_code_pilot_input("/nonexistent/path/file.json")

    def test_invalid_json_structure(self, tmp_path):
        """Test ValueError for non-dict top-level JSON."""
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(["not", "a", "dict"]))

        with pytest.raises(ValueError, match="not valid JSON"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_missing_required_field_name(self, valid_input_data, tmp_path):
        """Test missing 'name' field."""
        del valid_input_data["name"]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Missing required field: 'name'"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_missing_required_field_version(self, valid_input_data, tmp_path):
        """Test missing 'version' field."""
        del valid_input_data["version"]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Missing required field: 'version'"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_missing_required_field_evaluation_version(self, valid_input_data, tmp_path):
        """Test missing 'Evaluation_Version' field."""
        del valid_input_data["Evaluation_Version"]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Missing required field: 'Evaluation_Version'"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_missing_required_field_code_list(self, valid_input_data, tmp_path):
        """Test missing 'code_list' field."""
        del valid_input_data["code_list"]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Missing required field: 'code_list'"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_code_list_not_list(self, valid_input_data, tmp_path):
        """Test 'code_list' is not a list."""
        valid_input_data["code_list"] = "not a list"
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="'code_list' must be a list"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_empty_code_list(self, valid_input_data, tmp_path):
        """Test empty code_list is valid."""
        valid_input_data["code_list"] = []
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        result = NISTInputReader.load_code_pilot_input(input_file)

        assert len(result.code_list) == 0

    def test_trial_not_dict(self, valid_input_data, tmp_path):
        """Test trial item is not a dict."""
        valid_input_data["code_list"] = ["not a dict"]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Failed to parse trial at index 0"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_trial_missing_trial_id(self, valid_trial_data, valid_input_data, tmp_path):
        """Test trial missing 'trial_id' field."""
        del valid_trial_data["trial_id"]
        valid_input_data["code_list"] = [valid_trial_data]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Missing required field: 'trial_id'"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_trial_missing_testing_import_statement(
        self, valid_trial_data, valid_input_data, tmp_path
    ):
        """Test trial missing 'testing_import_statement' field."""
        del valid_trial_data["testing_import_statement"]
        valid_input_data["code_list"] = [valid_trial_data]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(
            ValueError, match="Missing required field: 'testing_import_statement'"
        ):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_trial_missing_primary_method_name(
        self, valid_trial_data, valid_input_data, tmp_path
    ):
        """Test trial missing 'primary_method_name' field."""
        del valid_trial_data["primary_method_name"]
        valid_input_data["code_list"] = [valid_trial_data]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(
            ValueError, match="Missing required field: 'primary_method_name'"
        ):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_trial_missing_specification(
        self, valid_trial_data, valid_input_data, tmp_path
    ):
        """Test trial missing 'specification' field."""
        del valid_trial_data["specification"]
        valid_input_data["code_list"] = [valid_trial_data]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Missing required field: 'specification'"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_trial_missing_prompt_fixed(
        self, valid_trial_data, valid_input_data, tmp_path
    ):
        """Test trial missing 'prompt_fixed' field."""
        del valid_trial_data["prompt_fixed"]
        valid_input_data["code_list"] = [valid_trial_data]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Missing required field: 'prompt_fixed'"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_whitespace_stripping(self, valid_trial_data, valid_input_data, tmp_path):
        """Test that whitespace is stripped from trial fields."""
        valid_trial_data["testing_import_statement"] = "  from genai_code_file import add  "
        valid_trial_data["primary_method_name"] = "  add  "
        valid_trial_data["specification"] = "  def add(x: int) -> int: pass  "
        valid_trial_data["prompt_fixed"] = "  Write tests  "
        valid_input_data["code_list"] = [valid_trial_data]
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        result = NISTInputReader.load_code_pilot_input(input_file)

        trial = result.code_list[0]
        assert trial.testing_import_statement == "from genai_code_file import add"
        assert trial.primary_method_name == "add"
        assert trial.specification == "def add(x: int) -> int: pass"
        assert trial.prompt_fixed == "Write tests"

    def test_parse_trial_error_propagation(self):
        """Test that _parse_trial errors are properly caught and re-raised."""
        invalid_trial = {"trial_id": "test"}  # Missing required fields

        with pytest.raises(ValueError, match="Missing required field"):
            NISTInputReader._parse_trial(invalid_trial, "test.json")

    def test_second_trial_fails(self, valid_input_data, tmp_path):
        """Test error reporting when second trial fails to parse."""
        # Add a second invalid trial
        valid_input_data["code_list"].append(
            {"trial_id": "invalid"}  # Missing required fields
        )
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(valid_input_data))

        with pytest.raises(ValueError, match="Failed to parse trial at index 1"):
            NISTInputReader.load_code_pilot_input(input_file)

    def test_validate_input_data_directly(self, valid_input_data):
        """Test _validate_input_data static method directly."""
        result = NISTInputReader._validate_input_data(valid_input_data, "test.json")

        assert isinstance(result, InputData)
        assert result.name == "Test Input"

    def test_parse_trial_directly(self, valid_trial_data):
        """Test _parse_trial static method directly."""
        result = NISTInputReader._parse_trial(valid_trial_data, "test.json")

        assert isinstance(result, InputCodeList)
        assert result.trial_id == "00001_add"
        assert result.primary_method_name == "add"