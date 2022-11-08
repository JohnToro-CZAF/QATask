# QATask

**NOTE**: 
- Run `python setup.py install ; pip install -r requirements.txt` before doing anything else.
- Run `python setup.py install` again after making any changes in folder `drqa`.

## To do:
- [x] Clean wiki articles
- [x] Attaching ID for each wiki article
- [x] SQL retrieving according to ID
- [x] Retriever returns ID
- [ ] Build a reader

## Possible retrievers:
- [ ] KNN
- [x] TF-IDF
- [ ] BM25
- [ ] Elastic search
- [ ] Exact string matching + POS and NER feature-based search
- [ ] DPR = BERT trained on question+context_passage vietnamese embeddings + FAISS for searching

## How to create database sqlite
First, create a folder named `qatask/database/SQLDB` with a `__init__().py` iniside it.

Download and save ZaloAI's datasets:
- [wiki articles](https://dl-challenge.zalo.ai/e2e-question-answering/wikipedia_20220620_cleaned.zip) 
as `qatask/database/data_wiki_cleaned/wikipedia.jsonl`
- Train file as `qatask/database/datasets/train_merged_final.json`
- Test file as `qatask/database/datasets/test_sample.json`

Then run the following script:

```
python qatask/retriever/build_db.py qatask/database/datasets/data_wiki_cleaned/ \ 
                                    qatask/database/wikipedia_db/wikisqlite.db \
                                    --preprocess qatask/retriever/wiki_preprocess.py
```

Now can can try interact with it by using the script:
```
python qatask/retriever/test_connect_sqlite.py qatask/database/wikipedia_db/wikisqlite.db
```

## How to train TF-IDF
Create folder `saved_models` and run:
```
python qatask/retriever/build_tfidf.py  qatask/database/wikipedia_db/wikisqlite.db \
                                        saved_models/ \
                                        --tokenizer vnm --num-workers 4
```
Note: current available Vietnamese tokenizers are `vnm` or `pyvi`

The Document Retriever can now be used interactively with:
```
python qatask/retriever/interactive.py --model qatask/database/wikipedia_db/<model_name>.npz
```

