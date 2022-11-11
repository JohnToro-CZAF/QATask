import os
from qatask.reader.builder import build_reader
from qatask.retriever.builder import build_retriever
from qatask.postprocessing.builder import build_postprocessor
from qatask.database.builder import build_database
from qatask.tokenizers.builder import build_tokenizer
import omegaconf
import os
import os.path as osp
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser("ZaloAI")
    parser.add_argument("--sample-path", type=str, default="qatask/database/datasets/train_test_files/test_sample.json")
    parser.add_argument("--output-path", type=str, default="qatask/database/datasets/output/test_answer_submission.json")
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    return args

class Pipeline:
    def __init__(self, cfg) -> None:
        self.tokenizer = build_tokenizer(cfg.tokenizer)
        if cfg.database.rebuild:
            self.db = build_database(cfg.database)
        else:
            self.db = None
        self.reader = build_reader(cfg.reader, self.tokenizer, cfg.database.database_path)
        self.retriever = build_retriever(cfg.retriever, self.tokenizer, cfg.database.database_path)
        self.postprocessor = build_postprocessor(cfg.postprocessor)
        
    def __call__(self, set_questions) -> str:
        results = self.retriever(set_questions)
        results = self.reader(results)
        final_results = self.postprocessor(results)
        return final_results

def main() -> None:
    args = parse_arguments()
    cfg = omegaconf.OmegaConf.load(args.cfg)
    if cfg.pipeline.type == "default":
        zaloai_pipeline = Pipeline(cfg)
    with open(osp.join(os.getcwd(), args.sample_path)) as f:
        file = json.loads(f.read())
    data = file['data']

    ##Auto saving as json
    results = zaloai_pipeline(data)
    with open(osp.join(os.getcwd(), args.output_path), 'w') as f2:
        json.dump(results, f2, ensure_ascii=False, indent=4)
    ##Building automatic solution submission system below

if __name__ == "__main__":
    main()
    