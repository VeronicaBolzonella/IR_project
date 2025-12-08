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
from models.safe_evaluator import ClaimEvaluator
import models.safe_evaluator as safe_evaluator
from sklearn.metrics import roc_curve, roc_auc_score

def sum(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    m = min(len(a), len(b))
    return a[:m] + b[:m]

def min(a, b):
    return np.minimum(a, b)

def max(a, b):
    return np.maximum(a, b)

def avg(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (a + b) / 2

def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def create_augmented_prompt(prompt, docs):
    docs_prompt = " ".join(docs)
    return prompt + " " + docs_prompt

def compute_roc_analysis(final_answers, ensemble_funcs=None):
    """
    Computes ROC-AUC score for individual Uncertainty Estimates (UEs) 
    and their ensembles against binary safety scores.

    Args:
        final_answers (dict): A dictionary where values are dicts containing 
                              "ue1", "ue2", and "safe_scores" lists.
        ensemble_funcs (list, optional): A list of functions for combining ue1 and ue2.
                                        Defaults to None.

    Returns:
        dict: A dictionary of results containing 'roc_auc' for each method, 
              and the corresponding 'fpr', 'tpr', and 'thresholds'.
    """
    all_ue1 = []
    all_ue2 = []
    all_safe_scores = []

    results = {}

    for _, scores in final_answers.items():
        all_ue1.extend(scores["ue1"]) 
        all_ue2.extend(scores["ue2"]) 
        
        all_safe_scores.extend(scores["safe_scores"])

    ue1 = np.array(all_ue1)
    ue2 = np.array(all_ue2)
    safe_labels = np.array(all_safe_scores) 
    # 0 is correct and 1 is incorrect to match uncertainty
    safe_labels = 1 - safe_labels

    # UE1 Analysis
    fpr1, tpr1, thresholds1 = roc_curve(safe_labels, ue1)
    auc1 = roc_auc_score(safe_labels, ue1)
    results["ue1"] = {"roc_auc": float(auc1), "fpr": fpr1.tolist(), "tpr": tpr1.tolist(), "thresholds": thresholds1.tolist()}
    
    # UE2 Analysis
    fpr2, tpr2, thresholds2 = roc_curve(safe_labels, ue2)
    auc2 = roc_auc_score(safe_labels, ue2)
    results["ue2"] = {"roc_auc": float(auc2), "fpr": fpr2.tolist(), "tpr": tpr2.tolist(), "thresholds": thresholds2.tolist()}

    if ensemble_funcs is not None:
        for f in ensemble_funcs:
            ue_ensemble = f(ue1, ue2)
            fpr_e, tpr_e, thresholds_e = roc_curve(safe_labels, ue_ensemble)
            auc_e = roc_auc_score(safe_labels, ue_ensemble)
            name = getattr(f, "__name__", str(f))
            results[name] = {"roc_auc": float(auc_e), "fpr": fpr_e.tolist(), "tpr": tpr_e.tolist(), "thresholds": thresholds_e.tolist()}
    
    return results


def main():
    """
        Instantiates the model, creates the queries dictionary and ranks the documents 
        accrding to given index.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, required=True, help="Path to the queries jsonl file")
    parser.add_argument("--index", type=str, required=True, help="Path to the index folder")
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
    retreived_docs = reranker.rank(args.index, queries,fast=False)
    log(f"Ranking complete, computing UEs...")
    
    # Define model API
    os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"
    model ="openrouter/qwen/qwen-2.5-7b-instruct"
    
    safe = ClaimEvaluator(rater=model, fast=True)
    
    if args.index is not None:
        safe_evaluator.INDEX_PATH = args.index
        
    final_answers = {}
    count =0
    
    for qid, q in queries.items():
        count +=1
        if count < 13 and count > 24:
            break
        
        try:
            
            final_answers[qid] = {}
            
            # Create prompt combining a query and the retrieved documents
            prompt = create_augmented_prompt(q, retreived_docs[qid][0]) # for testing: using only one relevant document
            
            # Generate text with qwen and compute UEs for claims
            ue = generate_with_ue(prompt, model=model, seed=seed)  # model as str
            
            
            final_answers[qid]["ue1"] = [float(item[0]) for item in ue['normalized_truth_values'][0]]  
            final_answers[qid]["ue2"] = [float(item[1]) for item in ue['normalized_truth_values'][0]] 
            
            # print("UE values computed:", final_answers[qid]["ue1"])
            
            claims = ue['claims']
            # Runs safe model on each claim 
            safe_results = [safe(atomic_fact=claim) for claim in claims]
            
            # Converts safe output into numeric values
            safe_results_numeric = [- 1 if result["answer"] == None else 0 if "Not" in result["answer"] else 1 for result in safe_results]
            final_answers[qid]["safe_scores"] = safe_results_numeric

            with open("scores_results_13-.json", "w") as f:
                json.dump(final_answers, f, indent=2)
                
        except Exception as e:
            log(f"Error processing query {qid}: {e} \n Moving on to next query")
            continue

    log(f"Scores ready")

    # with open("scores_results.json", "w") as f:
    #     json.dump(final_answers, f, indent=2)

    rocs = compute_roc_analysis(final_answers=final_answers, ensemble_funcs=[sum, min, max, avg])
    print(rocs)

    with open("roc_results.json", "w") as f:
        json.dump(rocs, f, indent=2)
    

if __name__ == '__main__':
    main()