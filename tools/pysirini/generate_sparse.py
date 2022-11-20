import os
import os.path as osp
import subprocess
import omegaconf 
import argparse
import joblib
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser("ZaloAI")
    parser.add_argument("--cfg", type=str, default='configs/retriever/BM25.yaml')
    return parser.parse_args()

def sparse_generator(cfg):
    subprocess.call("sh tools/index_sparse.sh {} {} {} {}".format(cfg.corpus,
                            cfg.language, cfg.index_path , cfg.threads), shell=True)
def main():
    args = parse_arguments()
    sparse_generator(omegaconf.OmegaConf.load(args.cfg))

if __name__ == "__main__":
    main()