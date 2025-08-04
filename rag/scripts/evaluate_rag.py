import json
from query_data_pc import query_rag

# "r" for read
with open("rag/data/evaluation_questions.json", "r") as data:
    questions = json.load(data)