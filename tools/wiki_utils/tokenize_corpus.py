import json
import argparse
from CocCocTokenizer import PyTokenizer
import re
import string

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="datasets/wikicorpus/wiki.jsonl")
    parser.add_argument("--output-path", type=str, default="datasets/wikiarticle_retrieve/wiki_sirini.json")
    parser.add_argument("--test-mode", type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_path = args.data_path
    T = PyTokenizer(load_nontone_data=True)
    with open(args.output_path, "w") as g:
        # control variable to produce small length sliced corpus, in purpose of examining the results
        # control = 0 
        cnt = 0
        with open(data_path) as f:
            for line in f:
                # control += 1
                # if(control > 20):
                #     break
                # Parse document
                cnt += 1
                doc = json.loads(line)
                doc['text'] = " ".join(T.word_tokenize(doc['text'], tokenize_option=0))
                # doc['text'] = doc['text'].translate(str.maketrans('','',string.punctuation))
                # pattern = re.compile(r'\(|\)|\[|\]|\"|\'|\{|\}|\?|\!|\;|\=|\+|\*|\%|\$|\#|\@|\^|\&|\~|\`|\||\.|\,')
                pattern = re.compile(r'\(|\)|\[|\]|\"|\'|\{|\}|\=|\+|\*')
                doc['text'] = pattern.sub(' ', doc['text'])
                doc['text'] = re.sub(' +',' ', doc['text'])
                # res = []
                # for w in doc['text'].split(" "):
                #     if w not in STOPWORDS_VN:
                #         res.append(w)

                # restext = " ".join(res)
                temp =  {
                    "id": doc['id'],
                    "contents": doc['text'] + '\n'
                }
                json.dump(temp, g, ensure_ascii=False)
                g.write("\n")
    print("Successfully converted {} docuements to sirini format and tokenized, ready to run index sirini".format(cnt))              
if __name__ == "__main__":
    main()