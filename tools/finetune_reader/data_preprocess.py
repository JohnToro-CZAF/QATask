""" Convert data to MRC format """

import json
from tqdm import tqdm
import re
import os
from nltk import word_tokenize as lib_tokenizer
import nltk
nltk.download('punkt')

dict_map = dict({})

def word_tokenize(text):
    global dict_map
    words = text.split()
    words_norm = []
    for w in words:
        if dict_map.get(w, None) is None:
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"')
        words_norm.append(dict_map[w])
    return words_norm


def strip_answer_string(text):
    text = text.strip()
    while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] != '(' and text[-1] == ')' and '(' in text:
            break
        if text[-1] == '"' and text[0] != '"' and text.count('"') > 1:
            break
        text = text[:-1].strip()
    while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] == '"' and text[-1] != '"' and text.count('"') > 1:
            break
        text = text[1:].strip()
    text = text.strip()
    return text


def strip_context(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def handle_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_read:
        json_data = json.load(file_read)
    qa_data = json_data['data']
    norm_samples = []
    error = 0

    print("Start parsing data...")
    for item in tqdm(qa_data, total=len(qa_data), desc="Chunk {}".format(file_path)):
        raw_context = item['text']
        raw_question = item['question']

        if item['category'] == 'FULL_ANNOTATION':
            short_candidate = item['short_candidate']
            short_candidate_start = item['short_candidate_start']
            wiki_answer = item['answer']

            prev_context = strip_context(raw_context[:short_candidate_start])
            answer       = strip_answer_string(short_candidate)
            post_context = strip_context(raw_context[short_candidate_start + len(answer):])

            prev_context = ' '.join(word_tokenize(prev_context))
            answer       = ' '.join(word_tokenize(answer))
            post_context = ' '.join(word_tokenize(post_context))

            context = "{} {} {}".format(prev_context, answer, post_context).strip()
            question     = ' '.join(word_tokenize(raw_question))

            norm_samples.append({
                "context": context,
                "question": question,
                "answer_text": answer,
                "answer_start_idx": len("{} {}".format(prev_context, answer).strip()) - len(answer),
            })

        # elif item['category'] == 'FALSE_LONG_ANSWER':
        #     context  = ' '.join(word_tokenize(raw_context))
        #     question = ' '.join(word_tokenize(raw_question))
        #     norm_samples.append({
        #         "context": context,
        #         "question": question,
        #         "answer_text": '',
        #         "answer_start_idx": 0,
        #     })

    print(f"Parsing completed!")
    return norm_samples
    

def preprocess(cfg):
    dict_data_squad = handle_file(cfg.data_path)
    with open(cfg.mrc_path, 'w', encoding='utf-8') as file:
        for item in dict_data_squad:
            file.write("{}\n".format(json.dumps(item, ensure_ascii=False)))
    print("Total: {} samples".format(len(dict_data_squad)))


if __name__ == "__main__":
    # debugging purpose
    class Config:
        def __init__(self) -> None:
            self.data_path = os.path.join(os.getcwd(), 'datasets/train_test_files/train_sample.json')
            self.mrc_path  = os.path.join(os.getcwd(), 'datasets/data_for_finetuning/mrc_format_file.jsonl')
    
    config = Config()
    preprocess(config)