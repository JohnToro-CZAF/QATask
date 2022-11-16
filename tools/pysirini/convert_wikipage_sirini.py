import json
import argparse
import os.path as osp
import os
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../qatask/database/datasets/wikicorpus/wiki.jsonl")
    parser.add_argument("--output-path", type=str, required="../qatask/database/datasets/wikipage_post/page_sirini.txt")
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    if not osp.exists("/".join(args.output_path.split("/")[:-1])):
        os.makedirs("/".join(args.output_path.split("/")[:-1]), exist_ok=True)
    with open(args.output_path, "w") as g:
        with open(data_path) as f:
            for line in f:
                # Parse document
                doc = json.loads(line)
                temp = {
                    "id": doc['id'],
                    "contents": doc['title'] + " \n"
                }
                json.dump(temp, g)
                g.write("\n")
if __name__ == "__main__":
    main()