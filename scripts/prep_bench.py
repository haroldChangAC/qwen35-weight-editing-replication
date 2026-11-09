"""MATH-500 and GSM8K test in the authors' format (same fields as exp8 code/evaluate.py prepare)."""
import json, os
from datasets import load_dataset
B = os.path.expanduser("~/wsp/bench"); os.makedirs(B, exist_ok=True)
def dump(name, rows):
    with open(f"{B}/{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))
dump("math500", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                 for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
dump("gsm8k", [{"id": f"gsm8k_test_{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
               for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))])
