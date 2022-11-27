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
    if pred_answer is None:
        return int(truth_answer is None)
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
            truth_answer = truth['answer']
        else:
            continue
        if pred.get('ans_type', 0) > 0:
            continue
            # truth_answer = None
        pred_answer = pred['answer']
        em = em_scores.append(compute_em(pred_answer, truth_answer))
        f1 = f1_scores.append(compute_f1(pred_answer, truth_answer))

    return dict(em=np.mean(em_scores), f1=np.mean(f1_scores))

def get_answer_accuracy_multiple(preds, truths):
    tot = 0
    cnt = 0
    for pred, truth in zip(preds, truths):
        if truth['category'] == 'PARTIAL_ANNOTATION': continue
        if truth['category'] == 'FULL_ANNOTATION':
            truth_answer = truth['answer']
        else:
            continue
        tot += 1
        flag = 0
        pred_answer = pred['answer']
        for ans in pred_answer:
            cnt += int(ans.lower() == truth['answer'].lower())
            if ans.lower() == truth['answer'].lower():
                flag = 1
                break
        if flag == 0:
            print(pred['answer'][0], " x ", truth['answer'])
    return cnt/tot

def get_answer_recall_multiple_concat(preds, truths):
    tot = 0
    cnt = 0
    acc = 0
    for pred, truth in zip(preds, truths):
        if truth['category'] != 'FULL_ANNOTATION': continue
        if 'answer_concat' not in pred.keys(): continue
        tot += 1
        pred_answer = pred['answer_concat']
        if pred['answer'][0].lower() == truth['answer'].lower():
            acc += 1
        for ans in pred_answer:
            if ans[0].lower() == truth['answer'].lower():
                cnt += 1
                break
    return cnt/tot, acc/tot, tot

def get_recall_reader(preds, truths) -> float:
    tot = 0
    cnt = 0
    for pred, truth in zip(preds, truths):
        if truth['category'] != 'FULL_ANNOTATION': continue
        if 'according_wikipages' in pred.keys():
            tot += 1
            pred_answer = pred['according_wikipages'][:1]
            for ans in pred_answer:
                if ans == truth['answer']:
                    cnt += 1
                    break
    return cnt/tot, tot

def get_retrieving_recall(pred, truth) -> float:
    # compare pred top_k and truth and return recall
    match = 0
    t = 0
    emp = 0
    for idx, question in enumerate(pred):
        truth[idx]['title'] = truth[idx]['title'].replace('( ', '(').replace(' )', ')')
        if '(định hướng)' in truth[idx]['title']:
            truth[idx]['title'] = truth[idx]['title'].replace('(định hướng)', '')
        if '_' in question['question']:
            question['question'] = question['question'].replace('_', ' ')
        if question['question'] == truth[idx]['question']:
            flag = 0
            truth[idx]['title'] = truth[idx]['title'].strip()
            if truth[idx]['title'] == '':
                emp += 1
                continue
            for candidate in question['candidate_wikipages']:
                if candidate == 'wiki/' + truth[idx]['title'].replace(" ","_"):
                  flag = 1
                  match += 1
                  break
            if flag == 0:
                t += 1
                
    print("failed to retrieve: {} questions".format(t), "while there are {} empty title".format(emp))
    return match/(len(pred) - emp)

def main(args):
    # open json file and read the data into a variable
    pred = None
    truth = None
    with open(args.pred) as fp:
        pred = json.load(fp)['data']
    with open(args.truth) as fp:
        truth = json.load(fp)['data']
    
    truth = [item for item in truth if item['category'] == 'FULL_ANNOTATION']
    assert len(truth) > len(pred)
    truth = truth[:len(pred)]

    print("Answer Accuracy: ", get_answer_accuracy_multiple(pred, truth))
    recall_reader, total_wiki = get_recall_reader(pred, truth)
    print("Recall of reader: {}, over {}".format(recall_reader, total_wiki))
    print("Recall when retrieving top_k: ", get_retrieving_recall(pred, truth))
    recall, acc, total_re_reading = get_answer_recall_multiple_concat(pred, truth)
    print("Recall when rereading : {} and accuracy {}, over {}".format(recall, acc, total_re_reading))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, default="datasets/output/BM25_BERT_val.json")
    parser.add_argument('--truth', type=str, default="datasets/train_test_files/train_sample.json")
    args = parser.parse_args()
    main(args)