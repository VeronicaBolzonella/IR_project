# Get the queries
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from pyserini.search.lucene import LuceneSearcher

class Reranker():
    def __init__(self, searcher):
        self.encoder = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rank(self, index, queries):
        self.encoder.eval()
        searcher = LuceneSearcher(index)
        results = {}

        for qid, q in queries.items():
            # First Pass BM25 (Lucene)
            hits = searcher.search(q, k=1000)
        
            # Scores per query and document id
            docs = []
            for hit in hits:
                doc = searcher.doc(hit.docid).raw()
                docs.append(doc)
        
            features = self.tokenizer(q, docs, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                score = self.encoder(**features).logits
            results[qid] = score
                    

        
model = Reranker()



queries = {}

with open('data/longfact-objects_celebrities.jsonl', 'r', encoding='utf-8') as f:
    id = 0
    for line in f:
        query = json.loads(line)
        id += 1
        queries[id] = query["prompt"]


# Lucene Searcher

searcher = LuceneSearcher('indexes/wiki_dump_index')

hits = searcher.search
for qid, q in queries.items():
        
        # First Pass BM25 (Lucene)
        hits = searcher.search(q, k=1000)
        
        # Scores per query and document id
        for i in range(len(hits)):
            res = [qid, hits[i].docid, hits[i].score]
        
        #print(res)
        
# Cross Encoder

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
model.eval()

results = {}

for qid, q in queries.items():
    features = tokenizer(q, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        score = model(**features).logits
    results[qid] = score

for qid, score in results.items():
    print(f"Query {qid}: score={score}")
