import json 
import re
# Read a json file
with open('datasets/wikiarticle_retrieve/wiki_sirini.json', 'r') as f:
# with open('datasets/wikicorpus/wiki.jsonl', 'r') as f:
    cnt = 0
    for line in f:
      l = json.loads(line)
      cnt += 1
      if cnt > 100 and cnt < 200:
        print('='*100)
        print(l['contents'])
        # print(l['title'])
        print('-'*100)
        # print(l['text'])
        # pattern = re.compile(r'\(|\)|\[|\]|\"|\'|\{|\}|\?|\!|\;|\=|\+|\*|\%|\$|\#|\@|\^|\&|\~|\`|\|')
        # l['contents'] = pattern.sub(' ', l['contents'])
        # print(re.sub(' +',' ', l['contents']))
    print(cnt)