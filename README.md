# Parallel-Sampling

A Python project for interacting with OpenAI's API to generate and evaluate structured responses using Pydantic models.

## Features

### Core Components

- **main.py**: Minimal program that sends a question to the OpenAI API and prints the response
- **utils.py**: Utility functions for structured sampling from OpenAI API:
  - `sample_str()`: Simple string response sampling
  - `sample()`: Structured JSON response sampling with Pydantic model validation
  - Includes retry logic with exponential backoff for reliability
- **idea.py**: DALY (Disability-Adjusted Life Year) idea generation system:
  - Generates seed approaches for improving DALY
  - Creates detailed idea proposals
  - Evaluates approach feasibility

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Simple Question-Answer
```bash
python main.py
```

### Structured Sampling
```bash
python utils.py
```

### DALY Idea Generation
```bash
python idea.py
```

## Requirements

- openai
- tenacity
- ipdb
- termcolor
- pydantic
