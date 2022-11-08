import os
import numpy as np
import torch 
import torch.nn as nn
from qatask.reader.builder import build_reader
from qatask.retriever.builder import build_retriever
from qatask.postprocessing import *
import argparse

def get_arguments():
    parser = argparse.ArgumentParser('QATask-ZaLoAI')
    args = parser.parse_args()
    return args

class QATaskPipeline:
    def __init__(self, args) -> None:
        self.reader = build_reader(**args.reader)
        self.retriever = build_retriever(**args.retriever)
        self.postprocessor = None
        
    def __call__(self, question) -> str:
        passage_ctx = self.reader(question)
        answer_raw = self.reader(passage_ctx)
        return self.postprocessor(answer_raw)

def main():
    args = get_arguments()
    pipeline = QATaskPipeline(args)
    ##Building automatic solution submission system below

if __name__ == "__main__":
    main()
    