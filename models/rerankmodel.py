# Get the queries
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from pyserini.search.lucene import LuceneSearcher

class Reranker():
    def __init__(self):
        self.encoder = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder.to(self.device).eval()

    def rank(self, index, queries):

        searcher = LuceneSearcher(index)
        results = {}

        for qid, q in queries.items():
            print("Query number, ", qid)
            print("Query text: ", q)

            # First Pass BM25 (Lucene)
            hits = searcher.search(q, k=1000)
            
            # Scores per query and document id
            docids = [h.docid for h in hits]
            docs = [self._extract_text(searcher,d) for d in docids]
        
            features = self.tokenizer(
                [q]* len(docs), 
                docs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
                ).to(self.device)
            
            # inference_mode doesn't compute gradients and renders it impossible to re-enable them 
            with torch.inference_mode():
                logits = self.encoder(**features).logits
            print("Shape of the logits: ", logits.shape)

            # ([1000,1]) to ()
            logits = logits.squeeze(-1)
            # print("Shape after squeezing: ", logits)
            top_i = torch.topk(logits, k=3).indices.tolist()
            top_docs = [docids[i] for i in top_i]
            top_scores = [logits[i].item() for i in top_i]
            print(f"Top 3 docs and scores: \n{top_docs} \n{top_scores}")
            results[qid] = list(zip(top_docs, top_scores))

        return results

                    
    def _extract_text(self, searcher, docid):
        doc = searcher.doc(docid)
        if doc is None:
            return ""
        # Raw because of setting in indexing.sh
        return doc.raw()



model = Reranker()

queries = {}

with open('data/longfact-objects_gaming.jsonl', 'r', encoding='utf-8') as f:
    id = 0
    for line in f:
        query = json.loads(line)
        id += 1
        queries[id] = query["prompt"]

model.rank("indexes/wiki_dump_index", queries)



# # Lucene Searcher

# searcher = LuceneSearcher('indexes/wiki_dump_index')

# hits = searcher.search
# for qid, q in queries.items():
        
#         # First Pass BM25 (Lucene)
#         hits = searcher.search(q, k=1000)
        
#         # Scores per query and document id
#         for i in range(len(hits)):
#             res = [qid, hits[i].docid, hits[i].score]
        
#         #print(res)
        
# # Cross Encoder

# model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
# tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
# model.eval()

# results = {}

# for qid, q in queries.items():
#     features = tokenizer(q, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         score = model(**features).logits
#     results[qid] = score

# for qid, score in results.items():
#     print(f"Query {qid}: score={score}")
