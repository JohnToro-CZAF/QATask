''' 
    Desc: Slicing each wiki article to passages by sliding window and stride to ensure maxium of 100 words/passage vs 256 tokens/passage.
    Source: https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-v2.md#document-collection-segmented
    Reference script: https://github.com/castorini/pyserini/blob/master/scripts/msmarco_v2/segment_docs.py
'''

import json
import argparse
import itertools
from tqdm import tqdm
from underthesea import sent_tokenize
from transformers import AutoTokenizer
from tools.wiki_utils import preprocess_segmenting
import textwrap, sys

tokenizer = AutoTokenizer.from_pretrained("nguyenvulebinh/vi-mrc-large")
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def create_segments(passages, max_length, stride):
    global tokenizer
    # splitting to extremely small sentences
    sentences = list(itertools.chain.from_iterable([sent_tokenize(p) for p in passages]))
    sentences = list(itertools.chain.from_iterable([sent.split("<comma>") for sent in sentences]))
    sentences = [sent.strip() for sent in sentences]

    # remove errors
    sentences = [sent for sent in sentences if len(sent) > 1]
    sentences = [sent+"," if sent[-1] != '.' else sent for sent in sentences]

    # combine into segments
    # TODO: alternative approach to split instead of tokenizing everytime
    segments = []
    # for sentence in sentences:
    #     if len(tokenizer.encode(sentence)) >= 512:
    #         print(sentence)
    #         sys.exit()
    i = 0
    while i < len(sentences)-1:
        j, segment = i, ""
        while j < len(sentences) and len(tokenizer.encode(segment)) < 510:
            if len(tokenizer.encode(segment+" "+sentences[j])) >= 510: break
            segment = "{} {}".format(segment, sentences[j])
            j += 1

        if segment[-1] != '.':
            if segment[-1] == ',': segment = segment[:-1]
            segment = segment + "."

        if len(tokenizer.encode(segment)) > 512:
            print('-'*50)
            print(len(segment.split()))
            print('-'*50)
            print(textwrap.fill(segment, 140))
            print('-'*50)
            sys.exit()

        segments.append(segment)
        i += 2 if j < 2 else j//2

    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="qatask/database/datasets/wikipedia.jsonl")
    parser.add_argument("--output-path", type=str, default="qatask/database/datasets/wikicorpus/wiki_segmented.jsonl")
    parser.add_argument('--max-length', default=3, help='maximum sentence length per passage')
    parser.add_argument('--stride', default=2, help='the distance between each beginning sentence of passage in a document')
    parser.add_argument("--debug-mode", action='store_true')
    args = parser.parse_args()
    if args.debug_mode:
        args.output_path = args.output_path[:-6] + '_debug.jsonl'

    # Parsing and slicing
    dict_data_squad = []
    lines = open(args.data_path, 'r').readlines()
    id = 0
    for step, line in tqdm(enumerate(lines), total=len(lines)):
        if args.debug_mode and step+1 == 5: break

        doc = json.loads(line)
        if doc['title'] == "Trang Chính" or "(định hướng)" in doc['title']: continue
        text = preprocess_segmenting(doc['text'])
        passages = [t.strip() for t in text.split("<endl>")][1:]
        segments = create_segments(passages, args.max_length, args.stride)
        for seg_id, segment in enumerate(segments):
            dict_data_squad.append({
                "id": f"{doc['id']}#{seg_id}",
                "title": doc['title'],
                "text": "{}: {}".format(doc['title'], segment)
            })
            id += 1
    
    # Saving the sliced passages
    with open(args.output_path, 'w') as out_file:
        for item in dict_data_squad:
            out_file.write("{}\n".format(json.dumps(item, ensure_ascii=False)))
    print("Generated {} docuements".format(id+1))

if __name__ == "__main__":
    main()
