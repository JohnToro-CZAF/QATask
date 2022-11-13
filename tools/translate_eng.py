from qatask.retriever.tfidf.doc_db import DocDB as _DocDB

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group

import argparse
import sqlite3
import json
import os
import sys
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

class DocDB(_DocDB):
    def __init__(self, db_path, id_start=0, id_end=400000):
        super(DocDB, self).__init__(db_path)
        self.doc_ids = self.get_doc_ids()
        self.doc_text = [self.get_doc_text(id) for id in self.doc_ids]
        self.tup = [(id, text) for id, text in zip(self.doc_ids, self.doc_text)][id_start:id_end]
        self.__exit__()
        self.connection = None

    
    def __getitem__(self, index):
        return self.tup[index][0], " ".join(self.tup[index][1].split(" ")[:700])
    
    def __len__(self):
        return len(self.doc_ids)


def store_contents(gpu, save_path, dataloader, tokenizer, model, rank):
    rank_save_path = save_path[:-6] + f"_{rank}" + ".jsonl"
    with open(rank_save_path, "w") as fp:
        run_loop = enumerate(dataloader) if rank != 0 else tqdm(enumerate(dataloader))
        for iter, (doc_ids, vn_texts) in run_loop:
            outputs = model.module.generate(
                tokenizer(vn_texts, return_tensors="pt", padding=True).input_ids.to(gpu), 
                max_length=512
            )
            en_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for doc_id, en_text in zip(doc_ids, en_texts):
                temp = {
                    "id": str(doc_id),
                    "contents": en_text[4:] + "\n"
                }
                json.dump(temp, fp)
                fp.write("\n")

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    init_process_group(backend="nccl", init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    tokenizer = AutoTokenizer.from_pretrained("VietAI/envit5-translation")
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation").cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    doc_db  = DocDB(args.db_path, args.id_start, args.id_end)
    sampler = DistributedSampler(doc_db, num_replicas=args.world_size, rank=rank)
    batch_size = args.effective_batch_size // torch.cuda.device_count()
    dataloader = DataLoader(doc_db, batch_size=batch_size, pin_memory=False, shuffle=False, sampler=sampler)
    store_contents(gpu, args.save_path, dataloader, tokenizer, model, rank)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=3)
    parser.add_argument('--nr', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--id-start', type=int, required=True, help="(0,600000")
    parser.add_argument('--id-end', type=int, required=True)
    parser.add_argument('--db-path', default="qatask/database/wikipedia_db/wikisqlite.db", type=str)
    parser.add_argument('--save-path', default="qatask/database/wikipedia_faiss/wikipedia_pyserini_format.jsonl", type=str)
    parser.add_argument('--effective-batch-size', type=int, default=27)
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    mp.spawn(train, nprocs=args.gpus, args=(args,))

    # Merge jsonl files
    with open(args.save_path, "w") as fp:
        for gpu in range(args.gpus):
            rank_save_path = args.save_path[:-6] + f"_{gpu}" + ".jsonl"
            with open(rank_save_path) as temp_fp:
                for line in temp_fp:
                    json.dump(json.loads(line), fp)
                    fp.write("\n")

if __name__ == '__main__':
    main()
    