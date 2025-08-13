# pip install datasets ragas python-dotenv pandas
import json
import argparse
import os
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

BATCH_SIZE = 6        # keep it simple; change here if needed
SLEEP_BETWEEN = 2.0   # seconds between batches

def batched(iterable, n=BATCH_SIZE):
    """Yield successive n-sized batches from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG dataset (question/answer/contexts JSON) with batching and save results to CSV in rag/eval_results folder.")
    parser.add_argument("eval_set_path", help="Path to JSON file with evaluation samples")
    parser.add_argument("--out_csv", default="ragas_results.csv", help="File name where results will be saved to e.g., naive_data_results.csv")
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--embed", default="text-embedding-3-small")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    # Load rows: [{"question": "...", "answer": "...", "contexts": ["...", ...]}, ...]
    rows = json.loads(Path(args.eval_set_path).read_text())
    if not rows:
        raise ValueError("Input JSON is empty.")

    # Prepare eval_results folder
    output_folder = Path("rag/eval_results")
    output_folder.mkdir(exist_ok=True)
    out_csv_path = output_folder / args.out_csv

    # Judge + embeddings
    judge = ChatOpenAI(model=args.llm, temperature=args.temperature, max_tokens=args.max_tokens)
    ragas_llm = LangchainLLMWrapper(judge)

    embed = OpenAIEmbeddings(model=args.embed)
    ragas_embed = LangchainEmbeddingsWrapper(embed)

    # Run in batches
    all_dfs = []
    total = len(rows)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Evaluating {total} samples in {num_batches} batches (size={BATCH_SIZE})")

    for bi, batch in enumerate(batched(rows), start=1):
        print(f"[batch {bi}/{num_batches}] {len(batch)} samples…")
        ds_batch = Dataset.from_list(batch)
        try:
            res = evaluate(
                ds_batch,
                metrics=[faithfulness, answer_relevancy],
                llm=ragas_llm,
                embeddings=ragas_embed
            )
            df_batch = res.to_pandas()  # per-sample scores
            all_dfs.append(df_batch)
        except Exception as e:
            print(f"[Error] Batch {bi} failed: {e}")
        time.sleep(SLEEP_BETWEEN)

    if not all_dfs:
        raise RuntimeError("No batches succeeded; no results to save.")

    # Merge all batch results
    df = pd.concat(all_dfs, ignore_index=True)

    # Save CSV in eval_results folder
    df.to_csv(out_csv_path, index=False)

    print(f"\nDone. Evaluated {len(df)} samples.")
    print(f"Wrote per-sample scores to {args.out_csv}")

if __name__ == "__main__":
    main()
