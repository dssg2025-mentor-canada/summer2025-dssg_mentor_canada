# pip install datasets ragas
from datasets import Dataset
import json
from pathlib import Path

from query_data_pc import run_query  # must return (answer: str, contexts: List[str])

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Load questions (supports [{"question": "..."}] or ["...", ...])
with open("rag/data/evaluation_questions.json", "r") as f:
    raw = json.load(f)
naive_questions = [q["question"] for q in raw] if raw and isinstance(raw[0], dict) else raw

# Build rows directly (robust across ragas versions)
rows = []
for q in naive_questions:
    answer, contexts = run_query(q)
    rows.append({
        "question": q,
        "answer": str(answer) if answer is not None else "",
        "contexts": [str(c) for c in (contexts or [])]
    })

# save rows to JSON (can evaluate later without re-running RAG)
out_path = Path("rag/data/naive_single_turn_samples.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(rows, f, indent=2)
print(f"Saved {len(rows)} samples to {out_path}")

# create Hugging Face Dataset and evaluate
dataset = Dataset.from_list(rows)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print("RAGAS (macro) results:", result)
