# QATask
## To do:
1. Clean wiki articles
2. Attaching ID for each wiki article
3. SQL retrieving according to ID
4. Retriever returns ID

## Possible retrievers:
1. KNN
2. TF-IDF
3. BM25
4. Elastic search
5. Exact string matching + POS and NER feature-based search
6. DPR = BERT trained on question+context_passage vietnamese embeddings + FAISS for searching

## How to create database sqlite
First create a folder for .db file named qatask/database/SQLDB then a file init.
Download the wiki articles: https://dl-challenge.zalo.ai/e2e-question-answering/wikipedia_20220620_cleaned.zip
And save it as path/to/data.
Then run the following script:
python build_db.py path/to/data qatask/database/SQLDB --preprocess wiki_preprocess.py
Now you can try interact with it by using script qatask/database/connecting.py Remember to replace your own path to .db