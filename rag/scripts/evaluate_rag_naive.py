# pip install datasets
from datasets import Dataset
import json

from query_data_pc import query_rag

from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy

# "r" for read
with open("rag/data/evaluation_questions.json", "r") as data:
    questions = json.load(data)

def build_single_turn_sample(question: str) -> SingleTurnSample:
    answer, c