import json 
import io
import re
cnt = 0
res = []
with open('wikipedia.jsonl') as f:
  for line in f:
    cnt = cnt+1
    doc = json.loads(line)
    wikipage = doc['title']
    wikipage = 'wiki/' + wikipage.replace(' ', '_')
    doc['text'] = re.sub(re.compile('(\nBULLET::::-)|(BULLET::::-)|(BULLET::::\d+)'), ' ', doc['text'])
    doc['text'] = re.sub(re.compile('\\"'), '', doc['text']) 
    pattern = re.compile('\n|=+')
    lines = re.split(pattern, doc['text'])
    lines = [line.strip() for line in lines]
    text = ' '.join(lines)
    text = text.replace('/^\s+|\s+$|\s+(?=\s)/g', ' ')
    print(text)
    res.append((doc['id'], text, wikipage))
    if(cnt > 100):
      break
with io.open('/home/ubuntu/hoang.pn200243/AQ/QATask/qatask/database/temp.json', 'w', encoding='utf-8') as f:
  json.dump(res, f, ensure_ascii=False)