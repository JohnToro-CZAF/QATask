import os
import os.path as osp
import subprocess
import hydra
from omegaconf import DictConfig
import argparse
import joblib
from tqdm import tqdm

class ProgressParallel(joblib.Parallel):
    ## Allowing for progress bar to be shown.
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def parse_arguments():
    parser = argparse.ArgumentParser("ZaloAI")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    return parser.parse_args()

def faiss_generator(cfg, shard_id, gpu_id):
    name_retriever = cfg.retriever
    if name_retriever == "colbert-v2":
        name_encoder = 'castorini/tct_colbert-v2-hnp-msmarco'
    elif name_retriever == "dpr":
        name_encoder = 'facebook/dpr-question_encoder-multiset-base'
    elif name_retriever == "ance":
        name_encoder = 'castorini/ance-msmarco-passage'

    basename = "checkpoint/indexes"
    savename = osp.join(basename, name_retriever)
    subprocess.call("sh tools/index.sh {} {} {} {} {} {} {}".format(cfg.corpus, shard_id, 
                        cfg.shard_num, savename, name_encoder, cfg.batch_size, gpu_id), shell=True)

def merge_faiss_shard(cfg):
    subprocess.call("python -m pyserini.index.merge_faiss_indexes --prefix {} --shard-num {}".format(cfg.retriever, cfg.shard_num))

@hydra.main(version_base=None, config_path="../configs/retriever", config_name="colbertv2")
def main(cfg : DictConfig) -> None:
    args = parse_arguments()
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    ProgressParallel(n_jobs=len(gpu_ids))(joblib.delayed(faiss_generator)(cfg, id, id) for id in gpu_ids)
    # merge_faiss_shard(cfg)

if __name__ == "__main__":
    main()