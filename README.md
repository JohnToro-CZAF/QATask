# QATask

**NOTE**: 
- Assume the ONLY working directory is `/absolute/path/to/QATask/`.
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
First, create a folder named `qatask/database/wikipedia_db` with a `__init__.py` iniside it.

Download and save ZaloAI's datasets:
- [wiki articles](https://dl-challenge.zalo.ai/e2e-question-answering/wikipedia_20220620_cleaned.zip) 
as `qatask/database/datasets/data_wiki_cleaned/wikipedia.jsonl`
- [Train and test files](https://dl-challenge.zalo.ai/e2e-question-answering/e2eqa-train+public_test-v1.zip) as `qatask/database/datasets/train_test_files/train_merged_final.json` and `qatask/database/datasets/train_test_files/test_sample.json`

Then run the following script:

```
python qatask/database/build_db.py  absolute/path/to/QATask/qatask/database/datasets/data_wiki_cleaned/ \
                                    absolute/path/to/QATask/qatask/database/wikipedia_db/wikisqlite.db \
                                    --preprocess qatask/database/wiki_preprocess.py --num-workers 32
```

Now can can try interact with it by using the script:
```
!python qatask/database/test_connect_sqlite.py qatask/database/wikipedia_db/wikisqlite.db
```

## How to train TF-IDF
Create folder `saved_models` and run:
```
python qatask/retriever/build_tfidf.py  absolute/path/to/QATask/qatask/database/wikipedia_db/wikisqlite.db \
                                        saved_models/ \
                                        --tokenizer vnm --num-workers 32
```
Note: current available Vietnamese tokenizers are `vnm` or `pyvi`

The Document Retriever can now be used interactively with:
```
python tools/TFIDF/interactive.py --model saved_models/<model_file>
```

## To create submission file
```
python tools/TFIDF/candidate_passage_gen.py --model saved_models/<model_file> \
                                            --tokenizer vnm
```
