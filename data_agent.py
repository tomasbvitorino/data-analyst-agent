"""
Data Analyst Agent
------------------
Takes a CSV and a natural-language question, uses a local LLM (via Ollama) to
generate pandas code that answers the question, executes it in a restricted
sandbox, and returns the result (plus a chart, if one was produced).

If the generated code errors out, the agent feeds the error back to the LLM
for one self-correction attempt before giving up — a simple but effective
agentic loop.
"""
import io
import contextlib
import re
import textwrap

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ollama


SYSTEM_PROMPT_TEMPLATE = """You are a data analyst assistant. You are given a pandas DataFrame called `df`.

Schema (column: dtype — example values):
{schema}

Rules:
- Write ONLY Python code inside a single ```python code block.
- The DataFrame is already loaded as `df` — do not re-read any CSV.
- Assign your final answer to a variable called `result` (a string, number, or pandas object).
- If a chart would help answer the question, create it with matplotlib and call `plt.savefig("output_chart.png")`. Otherwise don't create a chart.
- Do not import pandas or matplotlib — they are already imported as `pd`, `plt`.
- Keep the code short and focused on answering the question.
- Do not use any file, network, or system operations.
"""

FORBIDDEN_PATTERNS = [
    r"\bopen\(", r"\bos\.", r"\bsys\.", r"\bsubprocess\b", r"\beval\(",
    r"\bexec\(", r"\b__import__\b", r"\bimport\s+os\b", r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b", r"\bshutil\b", r"\brequests\b", r"\bsocket\b",
]


def describe_schema(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        sample = df[col].dropna().unique()[:3]
        lines.append(f"- {col}: {df[col].dtype} — e.g. {list(sample)}")
    return "\n".join(lines)


def extract_code(llm_response: str) -> str:
    match = re.search(r"```python\s*(.*?)```", llm_response, re.DOTALL)
    if not match:
        match = re.search(r"```\s*(.*?)```", llm_response, re.DOTALL)
    if not match:
        raise ValueError("No code block found in LLM response.")
    return textwrap.dedent(match.group(1)).strip()


def is_code_safe(code: str) -> bool:
    return not any(re.search(pattern, code) for pattern in FORBIDDEN_PATTERNS)


def run_code_sandboxed(code: str, df: pd.DataFrame):
    """Executes generated code with a restricted namespace. Returns (result, chart_path_or_None)."""
    plt.close("all")
    safe_globals = {
        "pd": pd,
        "plt": plt,
        "df": df,
        "__builtins__": {
            "len": len, "range": range, "sum": sum, "min": min, "max": max,
            "sorted": sorted, "list": list, "dict": dict, "set": set, "str": str,
            "int": int, "float": float, "round": round, "abs": abs, "enumerate": enumerate,
            "zip": zip, "print": print, "True": True, "False": False, "None": None,
        },
    }
    local_vars = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(code, safe_globals, local_vars)

    result = local_vars.get("result", stdout.getvalue().strip() or None)
    chart_path = "output_chart.png" if plt.get_fignums() or _chart_saved() else None
    return result, chart_path


def _chart_saved():
    import os
    return os.path.exists("output_chart.png")


class DataAnalystAgent:
    def __init__(self, csv_path: str, model: str = "llama3.1:8b"):
        self.df = pd.read_csv(csv_path)
        self.model = model
        self.schema = describe_schema(self.df)

    def _call_llm(self, messages):
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]

    def ask(self, question: str, max_retries: int = 1):
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(schema=self.schema)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        last_error = None
        for attempt in range(max_retries + 1):
            llm_response = self._call_llm(messages)
            try:
                code = extract_code(llm_response)
                if not is_code_safe(code):
                    raise ValueError("Generated code contains a forbidden operation.")
                result, chart_path = run_code_sandboxed(code, self.df)
                return {
                    "question": question,
                    "code": code,
                    "result": result,
                    "chart_path": chart_path,
                    "attempts": attempt + 1,
                }
            except Exception as e:
                last_error = e
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": f"That code raised an error: {e}. Please fix it and return the corrected code."
                })

        return {
            "question": question,
            "code": None,
            "result": f"Failed after {max_retries + 1} attempts. Last error: {last_error}",
            "chart_path": None,
            "attempts": max_retries + 1,
        }


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sales_data.csv"
    agent = DataAnalystAgent(csv_path)

    print(f"Data Analyst Agent ready. Loaded '{csv_path}' ({agent.df.shape[0]} rows, {agent.df.shape[1]} columns).")
    print("Ask a question about your data (or 'exit' to quit).\n")

    while True:
        question = input("> ")
        if question.strip().lower() in ("exit", "quit"):
            break
        response = agent.ask(question)
        print("\n--- Generated code ---")
        print(response["code"])
        print("\n--- Result ---")
        print(response["result"])
        if response["chart_path"]:
            print(f"\n[Chart saved to {response['chart_path']}]")
        print()
