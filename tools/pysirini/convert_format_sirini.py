import json
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../qatask/database/datasets/wikicorpus/wiki.jsonl")
    parser.add_argument("--output-path", type=str, default="../qatask/database/datasets/wikiarticle_retrieve/wiki_sirini.json")
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    with open(args.output_path, "w") as g:
        with open(data_path) as f:
            lines = f.readlines()
            for step, line in tqdm(enumerate(lines), total=len(lines)):
                doc = json.loads(line)
                temp = {
                    "id": doc['id'],
                    "contents": doc['text'] + " \n"
                }
                json.dump(temp, g)
                g.write("\n")

if __name__ == "__main__":
    main()