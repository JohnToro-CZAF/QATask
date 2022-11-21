''' 
    Desc: Slicing each wiki article to passages by sliding window and stride to ensure maxium of 100 words/passage.
    Reference: https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-v2.md#document-collection-segmented
'''

import json
import argparse
import itertools
from tqdm import tqdm
from underthesea import sent_tokenize, text_normalize
from tools.wiki_utils import preprocess_segmenting
import textwrap, sys


def create_segments(passages, title):
    # splitting to extremely small parts
    parts = []
    for p in passages:
        temp = []
        for sentence in sent_tokenize(p):
            temp.extend([part.strip() for part in sentence.split("<comma>")])
        parts.extend(temp)

    # remove outliers such as "." 
    parts = [part for part in parts if len(part) > 1]
    parts = [part+"," if part[-1] != '.' else part for part in parts]

    # combine into segments
    segments = []
    i = 0
    while i < len(parts)-1:
        j, segment = i, f"{title}:"
        while j < len(parts) and len(segment.split()) <= 100:
            if len(parts[j].split()) > 100:
                tokens = parts[j].split()
                adds = []
                while len(tokens) > 100:
                    adds.append(" ".join(tokens[:30]))
                    tokens = tokens[30:]
                adds.append(" ".join(tokens))
                parts = parts[:j] + adds + parts[j+1:]
            temp = "{} {}".format(segment, parts[j])
            if len(temp.split()) >= 100: break
            segment = temp
            j += 1   

        if segment[-1] != '.':
            if segment[-1] == ',': segment = segment[:-1]
            segment = segment + "."

        segments.append(segment)
        i += max(1, (j-i)//2)

    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="qatask/database/datasets/wikipedia.jsonl")
    parser.add_argument("--output-path", type=str, default="qatask/database/datasets/wikicorpus/wiki_segmented.jsonl")
    parser.add_argument("--debug-mode", action='store_true')
    args = parser.parse_args()
    if args.debug_mode:
        args.output_path = args.output_path[:-6] + '_debug.jsonl'

    # Parsing and slicing
    lines = open(args.data_path, 'r').readlines()
    out_file = open(args.output_path, 'w')
    id = 0
    for step, line in tqdm(enumerate(lines), total=len(lines)):

        if args.debug_mode and step+1 == 6200: break

        doc = json.loads(line)
        if doc['title'] == "Trang Chính" or "(định hướng)" in doc['title']: continue
        text = preprocess_segmenting(doc['text'])
        passages = [t.strip() for t in text.split("<endl>")][1:]   # exclude the first `single-title-pararaph`
        segments = create_segments(passages, doc['title'])
        for seg_id, segment in enumerate(segments):
            temp = {
                "id": str(id),
                "title": doc['title'],
                "text": segment,
            }
            out_file.write("{}\n".format(json.dumps(temp, ensure_ascii=False)))
            id += 1

    out_file.close()
    print("Generated {} documents".format(id+1))

if __name__ == "__main__":
    main()