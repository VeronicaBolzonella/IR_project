import json
import numpy as np
import sys
import os

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
                              "ue1", "ue2", and optionally "safe_scores" lists.
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

    for key, scores in final_answers.items():
        # Skip entries without safe_scores
        if "ue1" not in scores or "ue2" not in scores or "safe_scores" not in scores:
            print(f"Skipping {key} because it's missing required keys.")
            continue

        all_ue1.extend(scores["ue1"])
        all_ue2.extend(scores["ue2"])
        all_safe_scores.extend(scores["safe_scores"])

    # Convert to numpy arrays
    ue1 = np.array(all_ue1)
    ue2 = np.array(all_ue2)
    safe_labels = np.array(all_safe_scores)
    # 0 is correct and 1 is incorrect to match uncertainty
    safe_labels = 1 - safe_labels

    # UE1 Analysis
    fpr1, tpr1, thresholds1 = roc_curve(safe_labels, ue1)
    auc1 = roc_auc_score(safe_labels, ue1)
    results["ue1"] = {
        "roc_auc": float(auc1),
        "fpr": fpr1.tolist(),
        "tpr": tpr1.tolist(),
        "thresholds": thresholds1.tolist()
    }

    # UE2 Analysis
    fpr2, tpr2, thresholds2 = roc_curve(safe_labels, ue2)
    auc2 = roc_auc_score(safe_labels, ue2)
    results["ue2"] = {
        "roc_auc": float(auc2),
        "fpr": fpr2.tolist(),
        "tpr": tpr2.tolist(),
        "thresholds": thresholds2.tolist()
    }

    # Ensemble analysis
    if ensemble_funcs is not None:
        for f in ensemble_funcs:
            ue_ensemble = f(ue1, ue2)
            fpr_e, tpr_e, thresholds_e = roc_curve(safe_labels, ue_ensemble)
            auc_e = roc_auc_score(safe_labels, ue_ensemble)
            name = getattr(f, "__name__", str(f))
            results[name] = {
                "roc_auc": float(auc_e),
                "fpr": fpr_e.tolist(),
                "tpr": tpr_e.tolist(),
                "thresholds": thresholds_e.tolist()
            }

    return results

def main():

    input_folder = "data/results"
    output_file = "merged_results.json"

    merged_data = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                # Merge the content into the main dictionary
                merged_data.update(data)

    # Write the merged data to a new JSON file
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged {len(os.listdir(input_folder))} files into {output_file}")

    with open("merged_results.json", "r") as f:
        final_answers = json.load(f)

    # Check the keys
    print(final_answers.keys())

    rocs = compute_roc_analysis(final_answers=final_answers, ensemble_funcs=[sum, min, max, avg])

    with open("data/results/roc_results.json", "w") as f:
        json.dump(rocs, f, indent=2)
    

if __name__ == '__main__':
    main()