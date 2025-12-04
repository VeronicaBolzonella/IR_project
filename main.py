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

def compute_regression(final_answers, ensemble_funcs=None):
    all_sum_of_eigen = []
    all_semantic_entropy = []
    all_safe_scores = []

    results = {}

    for _, scores in final_answers.items():
        all_sum_of_eigen.extend(scores["sum_of_eigen"])
        all_semantic_entropy.extend(scores["p_true"])
        all_safe_scores.extend(scores["safe_scores"])

    # Convert to numpy arrays
    ue1 = np.array(all_sum_of_eigen)
    ue2 = np.array(all_semantic_entropy)
    safe = np.array(all_safe_scores)

    # Linear regression using np.polyfit
    slope1, intercept1 = np.polyfit(ue1, safe, 1)
    slope2, intercept2 = np.polyfit(ue2, safe, 1)

    coef1 = np.corrcoef(ue1, safe)[0,1]  
    coef2 = np.corrcoef(ue2, safe)[0,1] 

    results["sum_of_eigen"] = {"slope": slope1, "intercept": intercept1, "correlation": coef1}
    results["p_true"] = {"slope": slope2, "intercept": intercept2, "correlation": coef2}
    
    if ensemble_funcs is not None:
        for f in ensemble_funcs:
            ue_ensemble = f(ue1, ue2)
            print(ue_ensemble[:2])
            slope, intercept = np.polyfit(ue2, safe, 1)
            coef = np.corrcoef(ue_ensemble, safe)[0,1] 


            name = getattr(f, "__name__", str(f))
            results[name] = {"slope": slope, "intercept": intercept, "correlation": coef}

    return results


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
    model ="openrouter/qwen/qwen-2.5-7b-instruct"
    
    
    # Define safe - based on Luca's edited code
    #safe = ClaimEvaluator(rater=..., )
    
    # Make a dictionary like{ q1: {sum_of_eigen : [values], semantic_entropy : [values], safe_score: [values]}, q2 :... }
    
    final_answers = {}
    
    count = 0
    for qid, q in queries.items():
        if count > 0:
            break # for testing only one query
        count +=1
        
        final_answers[qid] = {}
        
        # Create prompt combining a query and the retrieved documents
        prompt = create_augmented_prompt(q, retreived_docs[qid][0]) # for testing: using only one relevant document
        
        # Generate text with qwen and compute UEs for claims
        ue = generate_with_ue(prompt, model=model, seed=seed)  # model will be just a string!
        
        print(ue)
        
        final_answers[qid]["sum_of_eigen"] = ue['normalized_truth_values'][0][0] # should save a whole list of values, NOT if its just for one text
        #final_answers[qid]["p_true"] = ue['normalized_truth_values'][0][1]
        
        # Gets claims for the generated text 
        claims = ue['claims']
        
        # Runs safe model on each claim 
        #safe_results = [safe(atomic_fact=claim) for claim in claims]
        
        # Converts safe output into numeric values
        #safe_results_numeric = [- 1 if result["answer"] == None else 0 if "Not" in result["answer"] else 1 for result in safe_results]
        #final_answers[qid]["safe_scores"] = safe_results_numeric
        
    log(f"Scores ready")

    # SAVE VALUES 
    with open("scores_results.json", "w") as f:
        json.dump(final_answers, f, indent=2)

    regs = compute_regression(final_answers=final_answers, ensemble_funcs=[sum, min, max, avg, havg])
    print(regs)

    with open("regression_results.json", "w") as f:
        json.dump(regs, f, indent=2)

    def sum(a, b):
        return a+b
    
    def min(a, b):
        return np.minimum(a, b)
    
    def max(a, b):
        return np.maximum(a, b)
    
    def avg(a, b):
        return (a+b)/2
    
    def havg(a, b):
        return 2/((1/a)+(1/b))


if __name__ == '__main__':
    main()
