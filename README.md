# Data Analyst Agent

A local LLM agent that answers natural-language questions about a CSV by writing and executing
its own pandas code — a "chat with your data" tool, running entirely offline.

## How it works

1. You give the agent a CSV and a question in plain English (e.g. *"Which category has the highest profit margin?"*)
2. The agent (Llama 3.1 8B, running locally via [Ollama](https://ollama.com)) is prompted with the
   DataFrame's schema and asked to write pandas code that computes the answer
3. The generated code runs in a **restricted sandbox** (no file/network/system access, limited builtins)
4. If the code raises an error, the agent feeds the error back to the LLM for **one self-correction attempt**
5. The result (and a chart, if one was generated) is returned to the user

This is a lightweight example of an **agentic loop**: plan (generate code) → act (execute) →
observe (catch errors) → retry — rather than a single LLM call.

## Why local/offline (Ollama)?

No API costs, no data leaves your machine — relevant for any real-world use case involving
sensitive business data. The trade-off is response quality/speed vs. a hosted frontier model,
which is called out explicitly in this README rather than hidden.

## Safety measures

- Regex-based blocklist rejects code using `os`, `sys`, `subprocess`, `eval`, `exec`, file I/O, or networking
- Execution uses a restricted `__builtins__` dict — only a safe subset of built-in functions is available
- The DataFrame is passed in-memory; the agent never re-reads or writes files itself

This is a reasonable safeguard for a portfolio/demo project, **not** a production-grade sandbox
(e.g. it doesn't protect against resource exhaustion or all forms of introspection). For a real
deployment, code execution should happen in an isolated container/process with resource limits.

## Setup

Requires [Ollama](https://ollama.com) running locally with a model pulled:
```bash
brew install ollama        # macOS
ollama serve &
ollama pull llama3.1:8b
```

Then:
```bash
pip install -r requirements.txt
```

## Usage

**CLI:**
```bash
python data_agent.py sales_data.csv
> Which category has the highest total revenue?
> What's the average discount by sales channel?
> exit
```

**Streamlit app** (upload your own CSV or use the bundled sample):
```bash
streamlit run app.py
```

## Example

```
> Which region has the highest average profit margin?

--- Generated code ---
df['margin'] = df['profit'] / df['revenue']
result = df.groupby('region')['margin'].mean().sort_values(ascending=False)

--- Result ---
region
North      0.421
Central    0.418
East       0.409
South      0.402
West       0.397
Name: margin, dtype: float64
```

## Sample dataset

`sales_data.csv` is a synthetic retail sales dataset (5,000 orders across 7 categories, 5
regions, 2 channels) generated with `generate_sample_data.py` — included so the project runs
out of the box without needing your own data.

## Stack

Python · pandas · matplotlib · Ollama (Llama 3.1 8B) · Streamlit

## Limitations

- Runs one question at a time — no multi-turn conversation memory (each question is independent)
- Local 8B model is noticeably weaker than GPT-4o/Claude at generating correct pandas code on the first try; the retry loop compensates for some of this
- Sandbox blocklist approach is defense-in-depth for a demo, not a hard security boundary
