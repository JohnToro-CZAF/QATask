import json
import argparse
from tools.wiki_utils import preprocess_slicing
import os
import os.path as osp

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../qatask/database/datasets/wikipedia.jsonl")
    parser.add_argument("--output-path", type=str, default="../qatask/database/datasets/wikicorpus/wiki.jsonl")
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    if not osp.exists("/".join(args.output_path.split("/")[:-1])):
        os.mkdir("/".join(args.output_path.split("/")[:-1]))
    with open(args.output_path, "w") as g:
        # control variable to produce small length sliced corpus, in purpose of examining the results
        # control = 0 
        with open(data_path) as f:
            id = 0
            for line in f:
                # control += 1
                # if(control > 10):
                #   break
                # Parse document
                doc = json.loads(line)
                doc['text'] = preprocess_slicing(doc['text'])
                lstpos = 0
                for pos, c in enumerate(doc['text']):
                  if c == "#":
                    temp = {
                        "id": str(id),
                        "title": doc['title'],
                        "text": doc["text"][lstpos:pos]
                    }
                    json.dump(temp, g, ensure_ascii=False)
                    g.write("\n")
                    id += 1
                    lstpos = pos + 1
                temp = {
                    "id": str(id),
                    "title": doc['title'],
                    "text": doc["text"][lstpos:]
                }
                json.dump(temp, g, ensure_ascii=False)
                g.write("\n")
                id += 1
                # print(id)
            print("Genrated {} docuements".format(id+1))              
if __name__ == "__main__":
    main()