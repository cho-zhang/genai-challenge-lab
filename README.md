# Overview
 Experimental pipelines for NIST GenAI challenges (https://ai-challenges.nist.gov/genai). Orchestrates LLM-based code or text generation with configurable prompts, multi-model support, and automated evaluations.

## Installation

```bash
# Download the repo
$ git clone git@github.com:cho-zhang/genai-challenge-lab.git && cd genai-challenge-lab

# Create a virtual environment named 'venv'
$ python -m venv venv
$ source venv/bin/activate

# Initialize the system and install dependencies locally
$ pip install -e .

# Verification of installation
$ python -c "import litellm; import tenacity; print('Environment Ready')"

```


## Results - 2025 NIST GenAI Code Pilot Challenge
- https://ai-challenges.nist.gov/code
- The custom prompts are only included in the System Description file.

| Submission Id | Model             | Hyperparameters                              | Custom Prompt Version | Prompt Number | Correct (%) | CI1 (%) | CIT (%) | Full Coverage Finds All Errors (%) |
|---------------|-----------------|---------------------------------------------|---------------------|---------------|------------|---------|---------|----------------------------------|
| 733           | gpt-4           | max_completion_tokens=500, reasoning_effort=None, temperature=0.0 | 2                   | 0             | 50.00      | 47.83   | 47.83   | 39.13                            |
|               |                 |                                             |                     | 1             | 65.21      | 54.34   | 60.86   | 41.30                            |
| 735           | gpt-4           | max_completion_tokens=500, reasoning_effort=None, temperature=0.0 | 3                   | 0             | 52.17      | 45.65   | 50.00   | 41.30                            |
|               |                 |                                             |                     | 1             | 71.73      | 60.86   | 65.21   | 41.30                            |
| 741           | gpt-5-mini      | max_completion_tokens=3000, reasoning_effort=minimal, temperature=0.0 | 4                   | 0             | 26.08      | 26.08   | 23.91   | 17.39                            |
|               |                 |                                             |                     | 1             | 65.21      | 58.69   | 65.21   | 54.34                            |
| 745           | claude-haiku-4-5 | max_completion_tokens=3000, reasoning_effort=None | 5                   | 0             | 34.78      | 32.60   | 34.78   | 21.73                            |
|               |                 |                                             |                     | 1             | 60.86      | 60.86   | 60.86   | 58.69                            |
| 753 (reupload 745) | claude-haiku-4-5 | max_completion_tokens=3000, reasoning_effort=None | 5                   | 0             | 34.78      | 32.60   | 34.78   | 21.73                            |
|               |                 |                                             |                     | 1             | 60.86      | 60.86   | 60.86   | 58.69                            |
| 754 (rerun)   | claude-haiku-4-5 | max_completion_tokens=3000, reasoning_effort=None | 5                   | 0             | 39.13      | 39.13   | 34.78   | 23.91                            |
|               |                 |                                             |                     | 1             | 58.69      | 58.69   | 58.69   | 54.34                            |
| 757           | gpt-5.2         | max_completion_tokens=4000, reasoning_effort=medium | 5                   | 0             | 43.47      | 43.47   | 43.47   | 39.13                            |
|               |                 |                                             |                     | 1             | 86.95      | 80.43   | 84.78   | 73.91                            |
