import json
import argparse
from tqdm import tqdm
from underthesea import word_tokenize, text_normalize
from tools.wiki_utils import preprocess_slicing
from traitlets.config.loader import ArgumentParser

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="qatask/database/datasets/wikipedia.jsonl")
    parser.add_argument("--output-path", type=str, default="qatask/database/datasets/wikicorpus/wiki.jsonl")
    parser.add_argument("--debug-mode", action='store_true')
    args = parser.parse_args()
    if args.debug_mode:
        args.output_path = args.output_path[:-6] + '_debug.jsonl'
    return args

def main():
    args = parse_arguments()
    dict_data_squad = []
    lines = open(args.data_path, 'r').readlines()
    id = 0
    for step, line in tqdm(enumerate(lines), total=len(lines)):
        if args.debug_mode:
            if step+1 == 10: break
        doc = json.loads(line)
        if doc['title'] == "Trang Chính" or "(định hướng)" in doc['title']: continue
        text = preprocess_slicing(doc['text'])
        passages = [t.strip() for t in text.split("<endl>")][1:]   # exclude the first `single-title-pararaph`
        for passage in passages:
            dict_data_squad.append({
                "id": str(id),
                "title": doc['title'],
                "text": "{}: {}".format(doc['title'], passage)
            })
            id += 1
    
    print("Generated {} docuements".format(id+1))
    with open(args.output_path, 'w') as out_file:
        for item in dict_data_squad:
            out_file.write("{}\n".format(json.dumps(item, ensure_ascii=False)))

if __name__ == "__main__":
    main()
