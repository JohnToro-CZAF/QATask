import json
import logging
import argparse
import tqdm
import os
from collections import namedtuple
from pathlib import Path
from itertools import islice
from typing import List, Dict, Tuple, Any, Iterator
from qatask.retriever.builder import build_retriever
from qatask.retriever.base import BaseRetriever
from ..get_accuracy import compute_f1

Retriever_cfg = namedtuple('Retriever_cfg', ['cfg', 'tokenizer', 'db_path'])

def wiki_to_title(wiki_title: str):
  return wiki_title[5:].replace('_', ' ')

def load_zalo_file(input_file: Path):
  if not os.path.exists(input_file):
    raise FileNotFoundError

  with open(input_file, 'r') as f:
    data = json.load(f)
  
  zalo_data = [question for question in data["data"] if question["category"] != "FALSE_LONG_ANSWER"]
  return zalo_data

def create_dpr_training_dataset(zalo_data: list, retriever: BaseRetriever, num_hard_negative_ctxs: int = 1):
  # Retrieve top-k passages for each question:
  zalo_data = retriever(zalo_data)
  for zalo_sample in zalo_data:
    hard_negative_ctxs = None
    if zalo_sample['category'] == "PARTIAL_ANNOTATION":
      hard_negative_ctxs = get_hard_negative_ctxs_f1(zalo_sample, num_hard_negative_ctxs)
    else: 
      hard_negative_ctxs = get_hard_negative_ctxs(zalo_sample, num_hard_negative_ctxs)
    positive_ctxs = [{"title": zalo_sample['title'], "text": zalo_sample['text']}]
    if not hard_negative_ctxs or not positive_ctxs:
        logging.error(
            f"No retrieved candidates for article {zalo_sample['passage_title']}, with question {zalo_sample['question']}"
        )
        continue
    dict_DPR = {
      "question" : zalo_sample['question'],
      "answers" : zalo_sample.get('short_candidate', ''),
      "positive_ctxs" : positive_ctxs,
      "negative_ctxs" : [],
      "hard_negative_ctxs" : hard_negative_ctxs
    }
    yield dict_DPR

def get_hard_negative_ctxs_f1(zalo_sample :dict, num_hard_negative_ctxs: int = 1):
  list_hard_neg_ctxs = []
  for ctx in zalo_sample['candidate_passages']:
    list_hard_neg_ctxs.append({"title": wiki_to_title(ctx[1]), "text": ctx[3], "passage_id":""})
  sorted(list_hard_neg_ctxs, key=lambda x: compute_f1(x['text'], zalo_sample['text']), reverse=True)
  return list_hard_neg_ctxs[1:2]

def get_hard_negative_ctxs(zalo_sample: dict, num_hard_negative_ctxs: int = 1):
  list_hard_neg_ctxs = []
  candidates_passages = zalo_sample['candidate_passages']
  for candidate_ctx in candidates_passages:
    # See more in returned retrieved passages qatask/rertriever/serini_retriever.py List((doc_id, wikipage, score, context))
    if zalo_sample['answer'].lower() in candidate_ctx[3].lower():
      # This is postitive case
      continue
    list_hard_neg_ctxs.append({"title": wiki_to_title(candidate_ctx[1]), "text": candidate_ctx[3], "passage_id":""})
  
  return list_hard_neg_ctxs[:num_hard_negative_ctxs]

def save_dataset(iter_dpr: Iterator, dpr_output_file: Path, total_nb_questions: int, split_dataset: bool):
  if split_dataset:
    nb_train_examples = int(total_nb_questions * 0.8)
    nb_dev_examples = int(total_nb_questions * 0.1)

    train_iter = islice(iter_dpr, nb_train_examples)
    dev_iter = islice(iter_dpr, nb_dev_examples)

    dataset_splits = {
      dpr_output_file.parent / f"{dpr_output_file.stem}_train.json": train_iter,
      dpr_output_file.parent / f"{dpr_output_file.stem}_dev.json": dev_iter,
      dpr_output_file.parent / f"{dpr_output_file.stem}_test.json": iter_dpr,
    }
  else:
    dataset_splits = {dpr_output_file: iter_dpr}

  for path, set_iter in dataset_splits.items():
    with open(path, "w", encoding="utf-8") as f:
      json.dump(list(set_iter), f, ensure_ascii=False, indent=4)

def main(args, retriever_cfg):
  print('Loading data from {}'.format(args.zalo_input_filename))
  # 1. Load file data:
  zalo_data = load_zalo_file(args.zalo_input_filename)

  # 2. Get retriever:
  retriever = build_retriever(retriever_cfg.cfg, retriever_cfg.tokenizer, retriever_cfg.db_path)

  # 3. Get DPR data:
  iter_DPR = create_dpr_training_dataset(zalo_data, retriever, args.num_hard_negative_ctxs)

  # 4. Save to file:
  save_dataset(iter_DPR, args.dpr_output_filename, len(zalo_data), args.split_dataset)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert a SQuAD JSON format dataset to DPR format.")
  parser.add_argument(
      "--zalo-input-filename",
      dest="zalo_input_filename",
      help="A dataset with a Zalo JSON format.",
      metavar="ZALO_in",
      default='datasets/train_test_files/train_sample.json',
  )
  parser.add_argument(
      "--dpr-output-filename",
      dest="dpr_output_filename",
      help="The name of the DPR JSON formatted output file",
      metavar="DPR_out",
      default="data_dpr/dpr.json",
  )
  parser.add_argument(
      "--num-hard-negative-ctxs",
      dest="num_hard_negative_ctxs",
      type=int,
      help="Number of hard negative contexts to use",
      metavar="num_hard_negative_ctxs",
      default=1,
  )
  parser.add_argument(
      "--split-dataset",
      dest="split_dataset",
      action="store_true",
      help="Whether to split the created dataset or not (default: False)",
  )
  args = parser.parse_args()
  args.dpr_output_filename = Path(args.dpr_output_filename)
  args.zalo_input_filename = Path(args.zalo_input_filename)
  class Config:
    def __init__(self) -> None:
        self.type = "bm25"
        self.top_k = 30
        self.index_path = "checkpoint/indexes/BM25"
  retriever_config = Retriever_cfg(Config(), None, 'qatask/database/wikipedia_db/wikisqlite.db')
  main(args, retriever_config)