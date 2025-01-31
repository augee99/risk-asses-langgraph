# Loan Decision LangGraph Project

This project creates agents to analyze credit report for loan decision

## What it does

The agent responsibility includes:

1. Takes a credit report PDF input
2. Generates key credit factors that critcal for loan appliaiton decision
3. Generates load application decision


## Getting Started

1. Create a `.env` file.

```bash
cp .env.example .env
```

2. Define required API keys in your `.env` file.
3. Customize the code in graph.py if using a different LLM than Sonnet3.5.


## How to customize

1. **Select a different model**: We default to Anthropic's Claude 3.5 Sonnet. Change the llm statement to intrdouce other LLMs.

## How to run

1. Python __init__.py