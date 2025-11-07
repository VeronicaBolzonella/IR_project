# Get the queries
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from pyserini.search.lucene import LuceneSearcher

class Reranker():
    """
        Reranker retrieval system. Ranks a corpus according to given queries and returns the 
        top three matches for each query according. Rank is performed using BM25 (Lucene) first,
        and reranking the top hits with an transformer-based cross-encoder. 

        Attributes:
            encoder (AutoModelForSequenceClassification): 
                The pretrained cross-encoder model used to compute relevance scores between queries and documents.
            tokenizer (AutoTokenizer): 
                The tokenizer corresponding to the encoder model, used to prepare query-document pairs.
            device (str): 
                The device used for inference ('cuda' if available, otherwise 'cpu').
            searcher (LuceneSearcher): 
                The Lucene searcher instance used for first-pass BM25 retrieval. Set in rank()

    """
    def __init__(self):
        self.encoder = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder.to(self.device).eval()

    def _extract_text(self, searcher, docid:int):
        """
            Extracts the raw text content of a document from a Lucene index, 
            given its document ID.

            Args:
                docid (int): id of document from which to extract text

            Returns:
                str: raw text of the document

            Note:
                The text is extracted to raw assuming the document was indexed with
                a raw field (eg --storeRaw)

        """
        doc = searcher.doc(docid)
        if doc is None:
            return ""
        # Raw because of setting in indexing.sh
        return doc.raw()


    def rank(self, index:str, 
            queries:dict[int, str], 
            first_pass:int = 1000
            )-> dict[str, list[tuple[str, float]]]:
        """
        Ranks the corpus index according to the queries using rerank (BM25 + Encoder). 

        Args:
            index (str): path to the index folder
            queries (dict[int, str]): dictionary of query id and query content 
            first_pass (int, optional): Number of top hits reranked by the Encoder. Defaults to 1000.

        Returns:
            dict[str, list[tuple[str, float]]]: Dictionary of queries and their top 3 documents. 
            return structure:
                {
                    query_id (str): [
                        (docid (str), score (float)),
                        (docid (str), score (float)),
                        (docid (str), score (float))
                    ],
                    ... 
                }
        """

        searcher = LuceneSearcher(index)
        results = {}

        for qid, q in queries.items():
            # First Pass BM25 (Lucene)
            hits = searcher.search(q, k=first_pass)
            
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

            # ([1000,1]) to ()
            logits = logits.squeeze(-1)
            # print("Shape after squeezing: ", logits)
            top_i = torch.topk(logits, k=3).indices.tolist()
            top_docs = [docids[i] for i in top_i]
            top_scores = [logits[i].item() for i in top_i]
            results[qid] = list(zip(top_docs, top_scores))

        return results

   
    