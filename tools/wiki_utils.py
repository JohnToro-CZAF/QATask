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

def preprocess_slicing(raw_text):
    text = re.sub(re.compile(r"\n\nBULLET::::-|\n\nBULLET::::|:\nBULLET::::-|:\n\nBULLET::::-|:\ ====\nBULLET::::-"), ": ", raw_text)
    text = re.sub(re.compile(r"\.\xa0\n\n-"), "; ", text)
    text = re.sub(re.compile(r"\n\n-|:\n\n-|:\n\n"), ": ", text)
    text = re.sub(re.compile(r"\n\n"), "<endl>", text)
    text = re.sub(re.compile(r"\. =\nBULLET::::-|\. ==\nBULLET::::-|\. ===\nBULLET::::-|\. ====\nBULLET::::-|\. =====\nBULLET::::-"), ": ", text)
    text = re.sub(re.compile(r"\. =\nBULLET::::|\. ==\nBULLET::::|\. ===\nBULLET::::|\. ====\nBULLET::::|\. =====\nBULLET::::"), ": ", text)
    text = re.sub(re.compile(r"\. =\n|\. ==\n|\. ===\n|\. ====\n|\. =====\n"), ": ", text)
    text = re.sub(re.compile(r"=|==|===|====|====="), "", text)
    text = re.sub(re.compile(r"\.\nBULLET::::-|\nBULLET::::-"), "; ", text)
    text = re.sub(re.compile(r"BULLET::::-|BULLET::::"), "", text)
    text = re.sub(re.compile(r"\n"), "; ", text)
    return text

def preprocess_segmenting(raw_text):
    text = re.sub(re.compile(r'[\u4e00-\u9fff]+'), " ", raw_text)  # range of chinese, japanese, korean
    text = re.sub(re.compile(r"\n\nBULLET::::-|\n\nBULLET::::|:\nBULLET::::-|:\n\nBULLET::::-|:\ ====\nBULLET::::-"), " <comma> ", text)
    text = re.sub(re.compile(r"\.\xa0\n\n-|;|,|<br>|-|—|–|\|"), " <comma> ", text)
    text = re.sub(re.compile(r"\n\n-|:\n\n-|:\n\n"), " <comma> ", text)
    text = re.sub(re.compile(r"\n\n"), " <endl> ", text)

    text = re.sub(re.compile(r"\. =\nBULLET::::-|\. ==\nBULLET::::-|\. ===\nBULLET::::-|\. ====\nBULLET::::-|\. =====\nBULLET::::-"), " <comma> ", text)
    text = re.sub(re.compile(r"\. =\nBULLET::::|\. ==\nBULLET::::|\. ===\nBULLET::::|\. ====\nBULLET::::|\. =====\nBULLET::::"), " <comma> ", text)
    text = re.sub(re.compile(r"\. =\n|\. ==\n|\. ===\n|\. ====\n|\. =====\n"), " <comma> ", text)
    text = re.sub(re.compile(r"=|==|===|====|====="), " ", text)
    text = re.sub(re.compile(r"\.\nBULLET::::-|\nBULLET::::-"), " <comma> ", text)
    text = re.sub(re.compile(r"BULLET::::-|BULLET::::"), "", text)
    text = re.sub(re.compile(r"\n"), " <comma> ", text)
    return text
