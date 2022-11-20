from transformers import Trainer, TrainingArguments, AutoTokenizer

from .data_preprocess import preprocess
from .train_valid_split import data_split
from .utils import prepare_data, data_collator, compute_metrics
from qatask.reader.bertreader.mrc_model import MRCQuestionAnswering

import torch
import omegaconf
import numpy as np
from datasets import load_metric
import os
import sys

import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    return args


def get_model(cfg):
    model = MRCQuestionAnswering.from_pretrained(cfg.checkpoint, cache_dir=cfg.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
    return model, tokenizer


def get_trainer(cfg, model, tokenizer, train_dataset, valid_dataset):
    training_args = TrainingArguments(output_dir=cfg.saved_dir,
                                      do_train=True,
                                      do_eval=True,
                                      num_train_epochs=cfg.epoch_nums,
                                      learning_rate=cfg.learning_rate,
                                      warmup_ratio=0.05,
                                      weight_decay=0.01,
                                      per_device_train_batch_size=cfg.batch_size,
                                      per_device_eval_batch_size=cfg.batch_size,
                                      gradient_accumulation_steps=1,
                                      logging_dir=cfg.logging_dir,
                                      logging_steps=200,
                                      label_names=['start_positions',
                                                   'end_positions',
                                                   'span_answer_ids',
                                                   'input_ids',
                                                   'words_lengths'],
                                      group_by_length=True,
                                      save_strategy="epoch",
                                      metric_for_best_model='f1',
                                      load_best_model_at_end=True,
                                      save_total_limit=2,
                                      #eval_steps=1,
                                      #evaluation_strategy="steps",
                                      evaluation_strategy="epoch",
                                      )

    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


def main() -> None:
    args = parse_arguments()
    cfg = omegaconf.OmegaConf.load(args.cfg)
    preprocess(cfg.preprocess)
    data_split(cfg.datasplit)
    train_dataset, valid_dataset = prepare_data(cfg.datasplit)
    
    model, tokenizer = get_model(cfg.model)
    if torch.cuda.is_available() and cfg.model.cuda: model.cuda()

    trainer = get_trainer(cfg.model, model, tokenizer, train_dataset, valid_dataset)
    trainer.train()


if __name__ == "__main__":
    main()