import json 
import re
# Read a json file
with open('training_datasets/combine.json', 'r') as f:
# with open('datasets/wikicorpus/wiki.jsonl', 'r') as f:
    cnt = 0
    data = json.loads(f.read())['data'][0:100]
    for idx, item in enumerate(data):
      if idx > 2:
        break
      else:
        print(item)
        # print('='*100)
        # print(l['contents'])
        # print(l['title'])
        # pattern = re.compile(r'\(|\)|\[|\]|\"|\'|\{|\}|\?|\!|\;|\=|\+|\*|\%|\$|\#|\@|\^|\&|\~|\`|\|')
        # l['contents'] = pattern.sub(' ', l['contents'])
        # print(re.sub(' +',' ', l['contents']))