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

    def rank(self, index, queries, fast=True):
        # If fast = True, only do the BM25 without the cross-encoder
        searcher = LuceneSearcher(index)
        
        results = {}
        
        if fast:
            for qid, q in queries.items():
                hits = searcher.search(q, k=3)
                docsid = [h.docid for h in hits]
                results[qid] = list(docsid)
            
            print("Fast ranking results: ", results)
            return results
    

        for qid, q in queries.items():
            print("Query number, ", qid)
            print("Query text: ", q)
            
            # First Pass BM25 (Lucene)
            hits = searcher.search(q, k=1000)
            
            # Scores per query and document id
            docids = [h.docid for h in hits]
            docs = [self._extract_text(searcher,d) for d in docids]

            # Batches 
            batch_size = 32
            all_logits = []
            
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                
                features = self.tokenizer(
                    [q]* len(batch_docs), 
                    batch_docs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                    ).to(self.device)
            
                # inference_mode doesn't compute gradients and renders it impossible to re-enable them 
                with torch.inference_mode():
                    batch_logits = self.encoder(**features).logits.squeeze(-1)
                    all_logits.append(batch_logits.detach().cpu())
                    
                #print("Shape of the logits: ", logits.shape)

            logits = torch.cat(all_logits)
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

