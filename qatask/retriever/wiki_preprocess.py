"""Preprocess function to filter/prepare Wikipedia docs."""

import regex as re
from html.parser import HTMLParser

def preprocess(article):
    wikipage = article['title']
    text = article['text']
    wikipage = 'wiki/' + wikipage.replace(' ', '_')
    text = re.sub(re.compile('(\nBULLET::::-)|(BULLET::::-)|(BULLET::::\d+)'), ' ',text)
    text = re.sub(re.compile('\\"'), '', text) 
    lines = re.split(re.compile('\n|=+'), article['text'])
    lines = [line.strip() for line in lines]
    text = ' '.join(lines)
    text = text.replace('/^\s+|\s+$|\s+(?=\s)/g', ' ')
    # Return doc with `id` set to `title`
    return {'id': article['id'], 'text': text, 'wikipage': wikipage}
