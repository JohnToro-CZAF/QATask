"""Preprocess function to filter/prepare Wikipedia docs."""

import regex as re
from html.parser import HTMLParser

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
def main():
    sentence = "ại cương tập 1 (Lâm Ngọc Thiềm)  2. Hóa học đại cương (Phạm Văn Nhiêu)  3. Hóa học các quá trình (Vũ Đăng Độ)   Liên kết ngoài.  BULLET::::- 3D Chem - Chemistry, Structures, and 3D Molecules BULLET::::- IUMSC - Đại học Indiana Molecular Structure Center "
    pre_process(sentence)

if __name__ == '__main__':
    main()