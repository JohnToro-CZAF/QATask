import argparse
import re
import json
import string
import collections
import numpy as np
import sys

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_em(pred_answer, truth_answer):
    return int(normalize_answer(truth_answer) == normalize_answer(pred_answer))

def compute_f1(pred_answer, truth_answer):
    pred_tokens = get_tokens(pred_answer)
    truth_tokens = get_tokens(truth_answer)
    common = collections.Counter(truth_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())

    if len(truth_tokens) == 0 or len(pred_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(truth_tokens == pred_tokens)
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_answer_accuracy(preds, truths):
    em_scores = []
    f1_scores = []

    for pred, truth in zip(preds, truths):
        if truth['category'] == 'PARTIAL_ANNOTATION': continue
        if truth['category'] == 'FULL_ANNOTATION':
            truth_answer = truth['short_candidate']
        else:
            truth_answer = ""
        pred_answer = pred['answer']
        em = em_scores.append(compute_em(pred_answer, truth_answer))
        f1 = f1_scores.append(compute_f1(pred_answer, truth_answer))

    return dict(em=np.mean(em_scores), f1=np.mean(f1_scores))

def get_wiki_accuracy(pred, truth) -> float:
    # compare pred and truth and return accuracy
    match = 0
    for idx, question in enumerate(pred):
        if question == truth[idx]:
            for candidate in truth[idx]['candidates']:
                if candidate == question['answer']:
                  match += 1
    return match/len(pred)

def main(args):
    # open json file and read the data into a variable
    pred = None
    truth = None
    with open(args.pred) as fp:
        pred = json.load(fp)['data']
    with open(args.truth) as fp:
        truth = json.load(fp)['data']
    
    if args.answer_acc:
        accuracy = get_answer_accuracy(pred, truth)
    else:
        accuracy = get_wiki_accuracy(pred, truth)
    print(accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, default="qatask/database/datasets/wikipedia.jsonl")
    parser.add_argument('--truth', type=str, required=True, default="qatask/database/datasets/wikipedia_ans.jsonl")
    parser.add_argument('--answer-acc', action='store_true')
    args = parser.parse_args()
    main(args)