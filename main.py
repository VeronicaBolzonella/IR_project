import json
import argparse
from itertools import islice

from models.rerankmodel import Reranker

def main():
    """
        Instantiates the model, creates the queries dictionary and ranks the documents 
        accrding to given index.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, required=True, help="Path to the queries jsonl file")
    parser.add_argument("--index", type=str, default=1, help="Path to the index folder")
    args = parser.parse_args()

    model = Reranker()

    # Get the Fact Score Bio queries -> used for ranking and qwen
    queries = {}
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in islice(f,50):
            query = json.loads(line)
            id = query["qid"]
            queries[id] = query["prompt"]

    # Rank documents based on the queries
    
    model.rank(args.index, queries,fast=True)


    

# Function to create prompts for Qwen (it will go in messages = roles:user, content: ...)
def create_prompt(query, retrieved_documents):
    prompt="Based on the following documents, answer the question below.\n"
    for i, doc in enumerate(retrieved_documents): # assumes this is a list of documents texts, not ids, fix later
        prompt+= f"Document {i+1}: {doc}\n"
    prompt+=f"Question: {query}\n Answer:"
    return prompt

if __name__ == '__main__':
    main()

# TODO: move this to README eventually
# example usage: 
# python3 main.py --queries 'data/factscore_bio.jsonl' --index "indexes/wiki_dump_index"
# if you get module models not found make sure to add your working directory to the python path:
# export PYTHONPATH="${PYTHONPATH}:~/path/to/this/project"  