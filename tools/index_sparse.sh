python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $1 \
  --language $2 \
  --index $3 \
  --generator DefaultLuceneDocumentGenerator \
  --threads $4 \
  --storePositions --storeDocvectors --storeRaw