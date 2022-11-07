# QATask

**NOTE**: Run `python -m pip install drqa` before doing anything else.

## To do:
- [x] Clean wiki articles
- [x] Attaching ID for each wiki article
- [x] SQL retrieving according to ID
- [ ] Retriever returns ID

## Possible retrievers:
- [ ] KNN
- [x] TF-IDF
- [ ] BM25
- [ ] Elastic search
- [ ] Exact string matching + POS and NER feature-based search
- [ ] DPR = BERT trained on question+context_passage vietnamese embeddings + FAISS for searching

## How to create database sqlite
First, create a folder named `qatask/database/SQLDB` with a `__init__().py` iniside it.

Download the [ZaloAI wiki articles](https://dl-challenge.zalo.ai/e2e-question-answering/wikipedia_20220620_cleaned.zip) and save it as `qatask/database/data_wiki_cleaned`.

Then run the following script:

```
python qatask/retriever/build_db.py qatask/database/datasets/data_wiki_cleaned/ \ 
                                    qatask/database/wikipedia_db/wikisqlite.db \
                                    --preprocess qatask/retriever/wiki_preprocess.py
```

Now can can try interact with it by using the script:
```
qatask/retriever/test_connect_sqlite.py qatask/database/wikipedia_db/wikisqlite.db
```

## How to train TF-IDF
Create a folder `saved_models/` then run:
```
python qatask/retriever/build_tfidf.py /qatask/database/wikipedia_db/wikisqlite.db \
                                       /saved_models/
```
