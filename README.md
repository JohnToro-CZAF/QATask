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
- [x] DPR = BERT trained on question+context_passage vietnamese embeddings + FAISS for searching

## How to create database sqlite
First, create a folder named `qatask/database/wikipedia_db` with a `__init__.py` iniside it.

Download and save ZaloAI's datasets:
- [wiki articles](https://dl-challenge.zalo.ai/e2e-question-answering/wikipedia_20220620_cleaned.zip) 
as `qatask/database/datasets/data_wiki_cleaned/wikipedia.jsonl`
- [Train and test files](https://dl-challenge.zalo.ai/e2e-question-answering/e2eqa-train+public_test-v1.zip) as `qatask/database/datasets/train_test_files/train_merged_final.json` and `qatask/database/datasets/train_test_files/test_sample.json`

Then run the following script:
If you want to use Sirini retrievers you need to translate Vietnamese corpus into english and in Sirini format
```
python3 -m tools.translate_eng 
```
Then you can create a FAISS index for your favourite Sirini retriever by configs file 
```
python3 tools/generating_dense.py --cfg configs/retriever/colbertv2.yaml 
``` 
Now you can have Sirini searcher as a normal retriever like TFIDF.  Just run `main` with your config `configs/colbertv2.yaml` 
```
python3 main.py --cfg configs/main/colbertv2.yaml
```
Or you can run TFIDF retriever baseline method which does not require any above command.
```
pytho3n main.py --cfg configs/main/baseline.yaml
```
If you want to add new modules. Please, visit qatask/* and inherit classes base.py. For example, 
```
XLMReader(BaseReader):
    def __init__(self, cfg)
        ...
```
and register your module class in `builder.py` and change your name class in baseline.yaml or make your own configuration file `your_config.yaml`
