import json
import argparse
from tools.wiki_utils.wiki_utils import preprocess_article

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="datasets/wikipedia.jsonl")
    parser.add_argument("--output-path", type=str, default="datasets/wikicorpus/wiki.jsonl")
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    with open(args.output_path, "w") as g:
        # control = 0
        with open(data_path) as f:
            id = 0
            for line in f:
                # control += 1
                # if(control > 30):
                #   break
                # Parse document
                doc = json.loads(line)
                doc['text'] = preprocess_article(doc['text'])
                temp = {
                    "id": str(id),
                    "title": doc['title'],
                    "text": doc["text"]
                }
                json.dump(temp, g, ensure_ascii=False)
                g.write("\n")
                id += 1  
            print("Genrated {} docuements".format(id+1))              
if __name__ == "__main__":
    main()