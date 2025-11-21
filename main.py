import json
import argparse
from itertools import islice
import numpy as np
import sys

from pyserini.search.lucene import LuceneSearcher

from models.rerankmodel import Reranker
from evaluation.ue import generate_with_ue

def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def create_augmented_prompt(prompt, docs):
    docs_prompt = " ".join(doc for doc in docs)
    return prompt + " " + docs_prompt

def main():
    """
        Instantiates the model, creates the queries dictionary and ranks the documents 
        accrding to given index.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, required=True, help="Path to the queries jsonl file")
    parser.add_argument("--index", type=str, default=1, help="Path to the index folder")
    args = parser.parse_args()

    log("Loading reranker model…")
    model = Reranker()

    log(f"Loading queries from {args.queries}…")
    queries = {}
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in islice(f,50):
            query = json.loads(line)
            id = query["qid"]
            queries[id] = query["prompt"]

    log("Ranking documents for queries…")
    retreived_docs = model.rank(args.index, queries,fast=True)
    log(f"Ranking complete, computing UEs...")
    
    sum_of_eigen = []
    semantic_entropy = []
    model_answers = []
    safe_scores = []
    for qid, q in queries.items():
        prompt = create_augmented_prompt(q, retreived_docs)
        ue = generate_with_ue(q, model=None, api=True)
        sum_of_eigen.append(ue['normalized_truth_values'][0])
        semantic_entropy.append(ue['normalized_truth_values'][1])
        model_answers.append(ue['generated_text'])
        # somehow get safe from the same answers
    log(f"Scores ready")

    # todo when 
    # safe_scores_numeric = prepare safe if needed
    # X = np.column_stack([sum_of_eigen, semantic_entropy])
    # y = np.array(safe_scores_numeric)

    # # Fit linear regression
    # reg = LinearRegression()
    # reg.fit(X, y)

    # print("Coefficients (sum_of_eigen, semantic_entropy):", reg.coef_)
    # print("Intercept:", reg.intercept_)


if __name__ == '__main__':
    main()
