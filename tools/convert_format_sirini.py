from qatask.preprocess.wiki_preprocess import preprocess_json
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--need-preprocess", type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    with open(args.output_path, "w") as g:
        with open(data_path) as f:
            for line in f:
                # Parse document
                doc = json.loads(line)
                if args.need_preprocess:
                    doc = preprocess_json(doc)
                temp = {
                    "id": doc['id'],
                    "contents": doc['text'] + " \n"
                }
                json.dump(temp, g)
                g.write("\n")
if __name__ == "__main__":
    main()