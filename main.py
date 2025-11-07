import json
import argparse

from models.rerankmodel import Reranker
from evaluation.ue import generate_with_ue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='what is the capital of Italy?', help="Prompt to model")
    
    args = parser.parse_args()

    output = generate_with_ue(args.prompt)
    print(output)


if __name__ == '__main__':
    main()