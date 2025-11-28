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

    seed = 42

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
    
    # Define safe - based on Luca's edited code
    safe = ClaimEvaluator(rater=..., )
    
    # Maybe make a dictionary like{ q1: [sum of eigen values, semantic entropy, safe_score]}
    sum_of_eigen = []
    semantic_entropy = []
    model_answers = []
    safe_scores = []
    
    
    for qid, q in queries.items():
        prompt = create_augmented_prompt(q, retreived_docs[qid])
        
        # Generate text with qwen and compute UEs for claims
        ue = generate_with_ue(prompt, model=None, api=True, seed=seed)  # model will be just a string!
        
        sum_of_eigen.append(ue['normalized_truth_values'][0])
        semantic_entropy.append(ue['normalized_truth_values'][1])
        
        # Gets claims for a generated text th
        claims = ue['claims']
        
        # Runs safe model on each claim 
        safe_results = [safe(atomic_fact=claim) for claim in claims]
        
        # Converts safe output into numeric values
        safe_results_numeric = [- 1 if result["answer"] == None else 0 if "Not" in result["answer"] else 1 for result in safe_results]
        
        
    log(f"Scores ready")

    # SAVE VALUES SOMEWHERE TO AVOID DISASTERS

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
