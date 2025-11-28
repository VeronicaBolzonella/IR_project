import json
import argparse
from itertools import islice
import numpy as np
import sys
import TruthTorchLM as ttlm
import os

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
    reranker = Reranker()

    log(f"Loading queries from {args.queries}…")
    queries = {}
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in islice(f,50):
            query = json.loads(line)
            id = query["qid"]
            queries[id] = query["prompt"]

    log("Ranking documents for queries…")
    retreived_docs = reranker.rank(args.index, queries,fast=True)
    log(f"Ranking complete, computing UEs...")
    
    # Define model API
    
    
    os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"
    model ="qwen/qwen2.5-7b-instruct"
    
    
    # Define safe - based on Luca's edited code
    #safe = ClaimEvaluator(rater=..., )
    
    # Make a dictionary like{ q1: {sum_of_eigen : [values], semantic_entropy : [values], safe_score: [values]}, q2 :... }
    
    final_answers = {}
    
    for qid, q in queries.items():
        if qid > 0:
            break # for testing only one query
        
        final_answers[qid] = {}
        
        # final_answers[qid]["sum_of_eigen"] = []
        # final_answers[qid]["semantic_entropy"] = []
        # final_answers[qid]["safe_scores"] = []
        
        # Create prompt combining a query and the retrieved documents
        prompt = create_augmented_prompt(q, retreived_docs[qid])
        
        # Generate text with qwen and compute UEs for claims
        ue = generate_with_ue(prompt, model=model, seed=seed)  # model will be just a string!
        
        print(ue)
        
        final_answers[qid]["sum_of_eigen"] = ue['normalized_truth_values'][0] # should save a whole list of values
        final_answers[qid]["semantic_entropy"] = ue['normalized_truth_values'][1]
        
        # Gets claims for the generated text 
        claims = ue['claims']
        
        # Runs safe model on each claim 
        #safe_results = [safe(atomic_fact=claim) for claim in claims]
        # Converts safe output into numeric values
        safe_results_numeric = [- 1 if result["answer"] == None else 0 if "Not" in result["answer"] else 1 for result in safe_results]
        final_answers[qid]["safe_scores"] = safe_results_numeric
        
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
