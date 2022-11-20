import json 

# Read a json file
# with open('datasets/wikiarticle_retrieve_trivialsliced/wiki_sirini.json', 'r') as f:
with open('datasets/wikipedia.jsonl', 'w') as g:
  with open('datasets/wikipedia_zalo.jsonl', 'r') as f:
    cnt = 0
    for line in f:
      l = json.loads(line)
      # print(l)
      if "định hướng" in l['title']:
        continue
      else:
        json.dump(l, g, ensure_ascii=False)
        g.write("\n")
        cnt += 1
  print(cnt)