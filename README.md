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
- [x] Build a reader
- [x] Properly slicing
- [x] Finetuned on ZaloAI dataset
- [x] Voting and combining retriever and reader scores
- [ ] Ensember 2 readers
- [ ] Other retrieving methods for vietnamese passages

## Possible retrievers:
- [x] TF-IDF
- [x] BM25
- [x] DPR = BERT trained on question+context_passage vietnamese embeddings + FAISS for searching
- [x] ANCE
- [x] ColBertv2

## How to create database sqlite
- First, create a folder named `qatask/database/wikipedia_db` with a `__init__.py` iniside it.
- Make an output folder by `mkdir output`
Download and save ZaloAI's datasets:
- [wiki articles](https://dl-challenge.zalo.ai/e2e-question-answering/wikipedia_20220620_cleaned.zip) 
as `datasets/zalo/wikipedia.jsonl`
- [Train and test files](https://dl-challenge.zalo.ai/e2e-question-answering/e2eqa-train+public_test-v1.zip) as `datasets/train_test_files/train_sample.json` and `datasets/train_test_files/test_sample.json`

To clean and slice the wiki articles, run:
```
python3 -m tools.wiki_slicing --data-path datasets/zalo/wikipedia.jsonl --output-path datasets/wikicorpus/wiki.jsonl
```

## BM25
Generate BM25 index. First, make `checkpoint/indexes/BM25` folder, then run this command to make BM25 index.

### Build retriever indexes
```
python3 -m tools.pysirini.convert_format_sirini --data-path datasets/wikicorpus/wiki.jsonl --output-path datasets/wikiarticle_retrieve/wiki_sirini.json

python3 -m tools.pysirini.generate_sparse --cfg configs/retriever/BM25.yaml
```

If you want to use BM25 post processor which retrieves wikipage as answer given a short candidate (produced by BERT), run this

### Build postprocessor indexes
```
python3 -m tools.pysirini.convert_wikipage_sirini --data-path datasets/zalo/wikipedia.jsonl --output-path datasets/wikipage_post/page_sirini.jsonl
              
python3 -m tools.pysirini.generate_sparse --cfg configs/postprocessor/BM25.yaml
```
### Noticing on postprocessor and retriever indexes and databases
since the databse that retriever and postprocessor used are different, retriever needs a sliced version of original corpus database, in meanwhile postprocessor's database is in need of original since the slicing made a lot of duplicates, leading to decrease in performance of getting the wiki links while the reader's performance is good. Here we only provide one database creating - for the retriever, reader (In configs/main/*.yml) but not for the postprocessor - you have to create one, or may be in the future we will add this into the pipeline.

### Running inference
After getting BM25 index, run main pipeline to output with finetuned BERT.
```
python3 main.py --cfg configs/main/BM25_finetunedBert.yaml \
                --sample-path datasets/train_test_files/test_sample.json \
                --output-path output/bm25_bert.json
```

## Faiss Retriever
Then run the following script:
If you want to use Sirini retrievers you need to translate Vietnamese corpus into english and in Sirini format
```
# translate Vietnamese corpus into english and change to Sirini format
python3 -m torch.distributed.launch -m tools.translate_eng

# Create a FAISS index for your favourite Sirini retriever by configs file 
python3 -m tools.pysirini.generating_dense.py --cfg configs/retriever/colbertv2.yaml 
``` 
Now you can have a Sirini searcher works like a normal retriever (e.g. TFIDF). Just run `main` with your config `configs/colbertv2.yaml`:
```
python3 main.py --cfg configs/main/colbertv2.yaml --output-path output/colbertv2_answer.json 
```

## Fine-tune reader
Change the parameters in the `configs/reader/xlm_roberta.yaml` file and run the following:
```
python3 tools/finetune_reader/train.py --cfg configs/reader/xlm_roberta.yaml
```
When the training completed, change the `reader.model_checkpoint` path in `configs/main/*.yaml` to the saved checkpoint.

## Main pipeline
Or you can run TFIDF retriever baseline method which does not require any above command.
```
python3 main.py --cfg configs/main/baseline.yaml
```

## Evaluate on the train dataset
By default, run this will calculate the EM of the predicted answers (wikipage, number, date):
```
python3 tools/get_accuracy.py --pred  output/<output_file>.json \
                              --truth datasets/train_test_files/train_sample.json
```
Optional: argument `--answer-acc` is used for calculating the F1 and EM of the short_candidate_answer, thus the `<output_file>.json` must be generated with out `postprocessing`.

## Customize
If you want to add new modules. Please, visit qatask/* and inherit classes base.py. For example, 
```
XLMReader(BaseReader):
    def __init__(self, cfg)
        ...
```
and register your module class in `builder.py` and change your name class in baseline.yaml or make your own configuration file `your_config.yaml`
