import json
import argparse
import os.path as osp
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="datasets/wikicorpus/wiki.jsonl")
    parser.add_argument("--output-path", type=str, default="datasets/wikiarticle_retrieve/wiki_sirini.json")
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    cnt = 0
    if not osp.exists(args.output_path):
        os.makedirs(osp.join(*args.output_path.split("/")[:-1]), exist_ok=True)
    with open(args.output_path, "w") as g:
        with open(data_path) as f:
            for line in f:
                cnt += 1
                # Parse document
                doc = json.loads(line)
                temp = {
                    "id": doc['id'],
                    "contents": doc['text'] + " \n"
                }
                json.dump(temp, g, ensure_ascii=False)
                g.write("\n")
    print("{} Documents are converted sirini format, ready to index".format(cnt))
if __name__ == "__main__":
    main()