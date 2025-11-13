import argparse
import json
from datetime import datetime
from pathlib import Path


from transformers import AutoTokenizer, AutoModelForCausalLM

from models import safe_evaluator

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, required=True, help="Path to the queries jsonl file")
    parser.add_argument("--index", type=str, default=None, help="Path to the index folder")
    args = parser.parse_args()

    # Overriding global index variable in safe_evaluator.py
    if args.index is not None:
        safe_evaluator.INDEX_PATH = args.index
        print("Using index path:", safe_evaluator.INDEX_PATH)

    queries = {}

    # This should be changed to a function to avoid repetition
    with open(args.queries, 'r', encoding='utf-8') as f:
        id = 0
        for line in f:
            query = json.loads(line)
            id += 1
            queries[id] = query["prompt"]


    # Loading model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda") 


    safe = safe_evaluator.ClaimEvaluator(rater=model,
            tokenizer = tokenizer,
            max_steps= 3,
            max_retries= 3,
            num_searches= 3,
            fast = True, # fast = True means BM25, else BM25+Cross-Encoder 
            )

    # Writing outputs to a new json file

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    output_path = Path("evaluation/outputs") / \
                f"safe_BM25_celebrities_{timestamp}.jsonl"

    # Check if evaluation/outputs exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for qid, q in queries.items():

            print(f"Evaluating query {qid}")

            result = safe(q)

            safe_outputs = {
                "id": qid,
                "prompt": q,
                "safe_answer": result["answer"],
                "safe_response": result["response"],
                "safe_search_details": result["search_details"]
            }

            f.write(json.dumps(safe_outputs) + "\n")

if __name__ == "__main__":
    main()


# For later: already implemented evaluation metrics
# sample_level_eval_metrics = ['f1']
# dataset_level_eval_metrics = ['auroc', 'prr']

# results = LFG.evaluate_truth_method_long_form(
#     dataset='longfact_objects',
#     model='gpt-4o-mini',
#     tokenizer=None,
#     sample_level_eval_metrics=sample_level_eval_metrics,
#     dataset_level_eval_metrics=dataset_level_eval_metrics,
#     decomp_method=decomposition_method,
#     claim_check_methods=[qa_generation],
#     claim_evaluator=safe,
#     size_of_data=3,
#     previous_context=[{'role': 'system', 'content': 'You are a helpful assistant. Give precise answers.'}], 
#     user_prompt="Question: {question_context}",
#     seed=41,
#     return_method_details=False,
#     return_calim_eval_details=False,
#     wandb_run=None,  
#     add_generation_prompt=True,
#     continue_final_message=False
# )


# Current job sh file:
"""
#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=output.out

source /vol/csedu-nobackup/course/I00041_informationretrieval/users/analeopold/IR_project/.venv/bin/activate

cd /vol/csedu-nobackup/course/I00041_informationretrieval/users/analeopold/IR_project

#python main.py --queries 'data/longfact-objects_celebrities.jsonl' --index "indexes/wiki_dump_index"

python models/qwen.py
"""