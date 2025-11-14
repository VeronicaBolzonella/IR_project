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

    queries = {}
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in islice(f,50):
            query = json.loads(line)
            id = query["qid"]
            queries[id] = query["prompt"]

    model.rank(args.index, queries,fast=True)


if __name__ == '__main__':
    main()

# TODO: move this to README eventually
# example usage: 
# python3 main.py --queries 'data/factscore_bio.jsonl' --index "indexes/wiki_dump_index"
# if you get module models not found make sure to add your working directory to the python path:
# export PYTHONPATH="${PYTHONPATH}:~/path/to/this/project"  