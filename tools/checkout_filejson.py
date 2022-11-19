import json 

# Read a json file
# with open('datasets/wikiarticle_retrieve_trivialsliced/wiki_sirini.json', 'r') as f:
with open('datasets/wikiarticle_retrieve/wiki_sirini.json', 'r') as f:
    cnt = 0
    for line in f:
      l = json.loads(line)
      print(l)
    print(cnt)