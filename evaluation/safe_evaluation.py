import json

from models.safe_evaluator import ClaimEvaluator
from models.rerankmodel import Reranker

queries = {}

# This should be changed to a function to avoid repetition
with open('data/longfact-objects_celebrities.jsonl', 'r', encoding='utf-8') as f:
    id = 0
    for line in f:
        query = json.loads(line)
        id += 1
        queries[id] = query["prompt"]


safe = ClaimEvaluator()