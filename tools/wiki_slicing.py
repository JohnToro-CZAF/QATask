import json
import argparse
from qatask.preprocess.wiki_preprocess import preprocess_json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    with open(args.output_path, "w") as g:
        # control = 0
        with open(data_path) as f:
            id = 0
            for idx, line in enumerate(f):
                # control += 1
                # if(control > 30):
                #   break
                # Parse document
                doc = json.loads(line)
                lstpos = 0
                cnt = 0
                for pos, c in enumerate(doc['text']):
                  if c == "\n":
                    cnt += 1
                  if cnt > 10:
                    cnt = 0
                    temp = {
                        "id": str(id),
                        "title": doc['title'],
                        "text": doc["text"][lstpos:pos]
                    }
                    temp = preprocess_json(temp)
                    if len(temp['text']) > 10:
                        json.dump(temp, g)
                        g.write("\n")
                        id += 1
                    # print(id)
                    # print(idx)
                temp = {
                    "id": id,
                    "title": doc['title'],
                    "text": doc["text"][lstpos:]
                }
                temp = preprocess_json(temp)
                if len(temp['text']) > 10:
                    json.dump(temp, g)
                    g.write("\n")
                id += 1
                print(id)
            print("Genrated {} docuements".format(id+1))              
if __name__ == "__main__":
    main()