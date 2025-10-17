# Get the queries
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from pyserini.search.lucene import LuceneSearcher

class Reranker():
    def __init__(self, model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device).eval()

    def rank(self, index, queries):
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
        
            features = self.tokenizer([q]*len(docs), 
                                      docs, 
                                      padding=True, 
                                      truncation=True, 
                                      return_tensors='pt').to(self.device)
            with torch.no_grad():
                scores = self.encoder(**features).logits

            top_indices = torch.topk(scores, k=3).indices
            top_docs = [docs[i] for i in top_indices]
            top_scores = scores[top_indices]

            results[qid] = list(zip(top_docs, top_scores.tolist()))
        
        return results

                  

        
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
