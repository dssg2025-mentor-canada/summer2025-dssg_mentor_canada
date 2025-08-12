# pip install datasets ragas
from datasets import Dataset
import json
import argparse
import os
from pathlib import Path

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# # pip install langchain_mistralai
# from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper

from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv

load_dotenv()

# Load questions (supports [{"question": "..."}] or ["...", ...])
def load_dataset(data: str):
    with open(data, "r") as f:
        raw = json.load(f)
    questions = [q["question"] for q in raw] if raw and isinstance(raw[0], dict) else raw

def main():
    parser = argparse.ArgumentParser(description="Evaluate an existing RAG evaluation dataset (question/answer/contexts JSON).")
    parser.add_argument("eval_set_path", help="Path to JSON file with evaluation samples")
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--embed", default="text-embedding-3-small")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    with open(args.eval_set_path, "r") as f:
        rows = json.load(f)

    # Expect rows like: [{"question": "...", "answer": "...", "contexts": ["...", "..."]}, ...]
    dataset = Dataset.from_list(rows)

    evaluator_llm = ChatOpenAI(model=args.llm,
                                temperature=args.temperature,
                                  max_tokens=args.max_tokens)
    ragas_llm = LangchainLLMWrapper(evaluator_llm)

    embed = OpenAIEmbeddings(model=args.embed)
    ragas_embed = LangchainEmbeddingsWrapper(embed)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embed,   # <- key change to avoid OpenAI embeddings
    )

    print(f"Evaluated {len(rows)} samples")
    print("RAGAS (macro) results:", result)

if __name__ == "__main__":
    main()