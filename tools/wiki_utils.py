"""Preprocess function to filter/prepare Wikipedia docs."""

import regex as re
from html.parser import HTMLParser
from underthesea import word_tokenize, text_normalize

def preprocess(article):
    wikipage = article['title']
    text = article['text']
    wikipage = 'wiki/' + wikipage.replace(' ', '_')
    text = re.sub(re.compile('(\nBULLET::::-)|(BULLET::::-)|(BULLET::::\d+)'), ' ',text)
    text = re.sub(re.compile('\\"'), '', text) 
    lines = re.split(re.compile('\n|=+'), text)
    lines = [line.strip() for line in lines]
    text = ' '.join(lines)
    text = text.replace('/^\s+|\s+$|\s+(?=\s)/g', ' ')
    # Return doc with `id` set to `title`
    return {'id': article['id'], 'text': text, 'wikipage': wikipage}

def pre_process(sentence):
    text = re.sub(re.compile('(\nBULLET::::-)|(BULLET::::-)|(BULLET::::\d+)'), ' ', sentence)
    text = re.sub(re.compile('\\"'), '', text) 
    lines = re.split(re.compile('\n|=+'), text)
    lines = [line.strip() for line in lines]
    text = ' '.join(lines)
    text = text.replace('/^\s+|\s+$|\s+(?=\s)/g', ' ')
    print(text)

def preprocess_slicing(text):
    text = re.sub(re.compile("\n\n"), "#", text)
    text = re.sub(re.compile(r"=\n|==\n|===\n|====\n|=====\n"), ' ', text)
    text = re.sub(re.compile(r"=|==|===|====|====="), '', text)
    text = re.sub(re.compile(r"BULLET::::-|BULLET::::"), '', text)
    text = re.sub(re.compile(r"\n"), '.#', text)
    text = re.sub(re.compile(r"\s+"), ' ', text)
    text = re.sub(re.compile(r"\.\s+"), '.#', text)
    return text