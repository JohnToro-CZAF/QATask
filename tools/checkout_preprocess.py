import json
import argparse
from qatask.preprocess.wiki_preprocess import preprocess_json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=30)
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    with open(args.output_path, "w") as g:
        # control = 0
        with open(data_path) as f:
            control = 0
            for idx, line in enumerate(f):
                control += 1
                if(control > args.threshold):
                  break
                # Parse document
                doc = json.loads(line)
                
if __name__ == "__main__":
    main()