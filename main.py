import json
import argparse
from itertools import islice

from pyserini.search.lucene import LuceneSearcher

from models.rerankmodel import Reranker
from evaluation.ue import generate_with_ue

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

    model = Reranker()

    queries = {}
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in islice(f,50):
            query = json.loads(line)
            id = query["qid"]
            queries[id] = query["prompt"]

    retreived_docs = model.rank(args.index, queries,fast=True)

    sum_of_eigen = []
    semantic_entropy = []
    for qid, q in queries.items():
        prompt = create_augmented_prompt(q, retreived_docs)
        ue = generate_with_ue(q, model=None, api=True)
        sum_of_eigen.append(ue['normalized_truth_values'][0])
        semantic_entropy.append(ue['normalized_truth_values'][1])


if __name__ == '__main__':
    main()
