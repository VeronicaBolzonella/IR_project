import numpy as np
import sys

def compute_regression_individual_q(final_answers, qid):
        sum_eig = np.array(final_answers[qid]["sum_of_eigen"])
        pt = np.array(final_answers[qid]["p_true"])
        safe = np.array(final_answers[qid]["safe_scores"])
        
        # Simple linear regression: slope and intercept
        slope_sum_eig, intercept_sum_eig = np.polyfit(sum_eig, safe, 1)
        slope_pt, intercept_pt = np.polyfit(pt, safe, 1)

        coef1 = np.corrcoef(sum_eig, safe)[0,1]  
        coef2 = np.corrcoef(pt, safe)[0,1]  


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
    final_answers = {
        1: {
            "sum_of_eigen": [0.8, 0.5, 0.9, 0.3],
            "p_true": [0.7, 0.6, 0.85, 0.4],
            "safe_scores": [1, 0, 1, -1]
        },
        2: {
            "sum_of_eigen": [0.2, 0.4, 0.6, 0.9],
            "p_true": [0.3, 0.5, 0.55, 0.8],
            "safe_scores": [0, 0, 1, 1]
        },
        3: {
            "sum_of_eigen": [0.1, 0.3, 0.7, 0.5],
            "p_true": [0.2, 0.4, 0.75, 0.6],
            "safe_scores": [-1, 0, 1, 0]
        }
    }

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
    
    regs = compute_regression(final_answers=final_answers, ensemble_funcs=[sum, min, max, avg, havg])
    print(regs)







if __name__ == '__main__':
    main()
