# pip install datasets ragas
from datasets import Dataset
import json
from pathlib import Path
import argparse

from query_data_pc import run_query  # must return (answer: str, contexts: List[str])

# Load questions (supports [{"question": "..."}] or ["...", ...])
def load_questions(path: str):
    with open(path, "r") as f:
        raw = json.load(f)
    return [q["question"] for q in raw] if raw and isinstance(raw[0], dict) else raw

def main():
    parser = argparse.ArgumentParser(description="Build RAG evaluation dataset")
    parser.add_argument("input_json", help="Path to the JSON file containing questions")
    parser.add_argument("output_json", help="Path to save the SingleTurnSamples JSON")
    args = parser.parse_args()

    # Build rows directly (robust across ragas versions)
    questions = load_questions(args.input_json)
    rows = []
    for q in questions:
        answer, contexts = run_query(q)
        rows.append({
            "question": q,
            "answer": str(answer) if answer is not None else "",
            "contexts": [str(c) for c in (contexts or [])]
        })

    # save rows to JSON (can evaluate later without re-running RAG)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {len(rows)} samples to {out_path}")

if __name__ == "__main__":
    main()


