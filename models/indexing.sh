
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/wiki_dump\
  --index indexes/wiki_dump_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 9 \
  --storePositions \
  --storeDocvectors \
  --storeRaw