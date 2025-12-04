# Get the queries
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from pyserini.search.lucene import LuceneSearcher

class Reranker():
    """
    Reranker model, retrieves the top 3 hits for a query using BM25 first and reranking with a Cross Encoder
    """
    def __init__(self):
        self.encoder = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder.to(self.device).eval()

    def rank(self, 
             index:str, 
             queries:str, 
             top_hits:int = 3, 
             top_bm25_hits:int = 1000, 
             fast=False, 
             batch_size:int = 32
            ):
        """
        Ranks the documents and returns the content of the top 3 hits.

        Args:
            index (str): path to the index folder.
            queries (str): path to the queries folder.
            top_hits (int): number of top documents to return. Defaults to 3.
            top_bm25_hits (int): number of top documents returned in the BM25 pass. Defaults to 1000.
            fast (bool, optional): if true the retriever only uses BM25 and does not rerank with the Cross
                encoder. Defaults to False.
            batch_size (int): Number of docs in a batch of the cross encoder pass. Defaults to 32.

        Returns:
            results (List(str)): list of the content of the top_hits documents
        """
        searcher = LuceneSearcher(index)
        
        results = {}
        
        if fast:
            # only BM25 pass
            for qid, q in queries.items():
                hits = searcher.search(q, k=top_hits)
                docids = [h.docid for h in hits]
                docs = [self._extract_text(searcher,d) for d in docids]  # raw text
                results[qid] = list(docs)
            
            return results

        for qid, q in queries.items():
            print("Query number, ", qid)
            print("Query text: ", q)
            
            # First Pass BM25 (Lucene)
            hits = searcher.search(q, k=top_bm25_hits)
            
            # Scores per query and document id
            docids = [h.docid for h in hits]
            docs = [self._extract_text(searcher,d) for d in docids]

            all_logits = []
            
            for i in range(0, len(docs), batch_size):
                # cross encoder pass
                batch_docs = docs[i:i + batch_size]
                features = self.tokenizer(
                    [q]* len(batch_docs), 
                    batch_docs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                    ).to(self.device)
            
                with torch.inference_mode():
                    batch_logits = self.encoder(**features).logits.squeeze(-1)
                    all_logits.append(batch_logits.detach().cpu())
                    

            logits = torch.cat(all_logits)
            top_i = torch.topk(logits, k=3).indices.tolist()
            top_docs = [docids[i] for i in top_i]
            
            docs = [self._extract_text(searcher,d) for d in top_docs]  # raw text
            results[qid] = list(docs)

        return results

                    
    def _extract_text(self, searcher, docid):
        # Returns document content for given document id
        doc = searcher.doc(docid)
        if doc is None:
            return ""
        # Raw because of setting in indexing.sh
        return doc.raw()
