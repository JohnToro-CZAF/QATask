import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import collections
import json
from tqdm.auto import tqdm
from nltk import word_tokenize
import torch.distributed as dist

tokenizer = AutoTokenizer.from_pretrained("nguyenvulebinh/vi-mrc-large")


def average_main(x, args):
    if not args.is_distributed:
        return x
    if args.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if args.is_main:
            x = x / args.world_size
    return x

def compute_metrics_phobart(args, metric, start_logits, end_logits, features, examples):

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue

                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > args.max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def compute_metrics(eval_pred, tokenizer=tokenizer):
    metric = datasets.load_metric("squad", cache_dir='./log/metric')
    # print(eval_pred)
    logits, labels = eval_pred
    logits = list(zip(logits[0], logits[1]))
    labels, span_ids, samples_input_ids, word_lengths = list(zip(labels[0], labels[1])), labels[2], labels[3], labels[4]
    predictions = []
    references = []
    for idx, (predict, span_truth, input_ids, sample_words_length) in enumerate(
            list(zip(logits, span_ids, samples_input_ids, word_lengths))):
        span_truth = np.delete(span_truth, np.where(span_truth == -100))
        input_ids = np.delete(input_ids, np.where(input_ids == -100))

        # Get the most likely beginning of answer with the argmax of the score
        answer_start = sum(sample_words_length[:np.argmax(predict[0])])
        # Get the most likely end of answer with the argmax of the score
        answer_end = sum(sample_words_length[:np.argmax(predict[1]) + 1])

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        answer_truth = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(span_truth))

        predictions.append({'prediction_text': answer, 'id': str(idx)})
        references.append({'answers': {'answer_start': [answer_start], 'text': [answer_truth]}, 'id': str(idx)})
    results = metric.compute(predictions=predictions, references=references)
    return results


def data_collator(samples):
    if len(samples) == 0:
        return {}

    for sample in samples:
        start_idx = sum(sample['words_lengths'][:sample['start_idx']])
        end_idx = sum(sample['words_lengths'][:sample['end_idx'] + 1])
        sample['span_answer_ids'] = sample['input_ids'][start_idx:end_idx]

    def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    input_ids = collate_tokens([torch.tensor(item['input_ids']) for item in samples], pad_idx=tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for i in range(len(samples)):
        attention_mask[i][:len(samples[i]['input_ids'])] = 1
    words_lengths = collate_tokens([torch.tensor(item['words_lengths']) for item in samples], pad_idx=0)
    answer_start = collate_tokens([torch.tensor([item['start_idx']]) for item in samples], pad_idx=0)
    answer_end = collate_tokens([torch.tensor([item['end_idx']]) for item in samples], pad_idx=0)
    span_answer_ids = collate_tokens([torch.tensor(item['span_answer_ids']) for item in samples],
                                     pad_idx=-100)

    batch_samples = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'words_lengths': words_lengths,
        'start_positions': answer_start,
        'end_positions': answer_end,
        'span_answer_ids': span_answer_ids
    }

    return batch_samples


def tokenize_function(example, tokenizer=tokenizer):
    example["question"] = example["question"].split()
    example["context"] = example["context"].split()
    # max_len_single_sentence = tokenizer.max_len_single_sentence
    max_len_single_sentence = 368

    question_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in example["question"]]
    context_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in example["context"]]
    valid = True
    if len([j for i in question_sub_words_ids + context_sub_words_ids for j in i]) > max_len_single_sentence - 1:
        question_ids = [j for i in question_sub_words_ids for j in i]
        context_ids = [j for i in context_sub_words_ids[:example['answer_word_end_idx'] + 1] for j in i]
        remain_tokens = max_len_single_sentence - 1 - len(question_ids)
        if len(question_ids + context_ids) < max_len_single_sentence - 1:
            context_sub_words_ids_revise = context_sub_words_ids[:example['answer_word_end_idx'] + 1]
            idx = example['answer_word_end_idx'] + 1
            while len([j for i in (context_sub_words_ids_revise + [context_sub_words_ids[idx]]) for j in
                       i]) < remain_tokens and idx < len(context_sub_words_ids):
                context_sub_words_ids_revise.append(context_sub_words_ids[idx])
                idx += 1
            context_sub_words_ids = context_sub_words_ids_revise
        else:
            valid = False

    question_sub_words_ids = [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
    if len(input_ids) > max_len_single_sentence + 2:
        valid = False

    words_lengths = [len(item) for item in question_sub_words_ids + context_sub_words_ids]

    return {
        "input_ids": input_ids,
        "words_lengths": words_lengths,
        "start_idx": (example['answer_word_start_idx'] + len(question_sub_words_ids)) if len(
            example["answer_text"]) > 0 else 0,
        "end_idx": (example['answer_word_end_idx'] + len(question_sub_words_ids)) if len(
            example["answer_text"]) > 0 else 0,
        "valid": valid
    }


def prepare_data(cfg):
    train_set = datasets.load_from_disk(cfg.train_path)
    valid_set = datasets.load_from_disk(cfg.valid_path)
    print("Train set: ", len(train_set))
    print("Valid set: ", len(valid_set))
    
    train_set = train_set.shuffle().map(tokenize_function, batched=False, num_proc=cfg.num_proc).filter(
        lambda example: example['valid'], num_proc=cfg.num_proc)
    valid_set = valid_set.map(tokenize_function, batched=False, num_proc=cfg.num_proc).filter(
        lambda example: example['valid'], num_proc=cfg.num_proc)

    print("Train set: ", len(train_set))
    print("Valid set: ", len(valid_set))
    return train_set, valid_set


if __name__ == "__main__":
    # debugging prepare_data
    class Config:
        def __init__(self) -> None:
            self.train_path = os.path.join(os.getcwd(), 'qatask/database/datasets/data_for_finetuning/train.dataset')
            self.valid_path = os.path.join(os.getcwd(), 'qatask/database/datasets/data_for_finetuning/valid.dataset')
            self.num_proc   = 10

    tokenizer = AutoTokenizer.from_pretrained("nguyenvulebinh/vi-mrc-large")
    train_dataset, valid_dataset = prepare_data(
        train_path=os.path.join(os.getcwd(), 'qatask/database/datasets/data_for_finetuning/train.dataset'),
        valid_path=os.path.join(os.getcwd(), 'qatask/database/datasets/data_for_finetuning/valid.dataset'),
    )
    from tqdm import tqdm

    for batch in tqdm(train_dataset):
        print("="*80)
        print(f"batch['context'] = {batch['context']}")
        print(f"batch['question'] = {batch['question']}")
        print(f"batch['answer_text'] = {batch['answer_text']}")
        print(f"batch['answer_start_idx'] = {batch['answer_start_idx']}")
        print(f"batch['answer_word_start_idx'] = {batch['answer_word_start_idx']}")
        print(f"batch['answer_word_end_idx'] = {batch['answer_word_end_idx']}")
        print(f"batch['input_ids'] = {batch['input_ids']}")
        print(f"batch['words_lengths'] = {batch['words_lengths']}")
        print(f"batch['start_idx'] = {batch['start_idx']}")
        print(f"batch['end_idx'] = {batch['end_idx']}")
        print(f"batch['valid'] = {batch['valid']}")
        break
        # print(batch['tgt_words_ids'].shape)

    for batch in tqdm(valid_dataset):
        print("="*80)
        print(f"batch['context'] = {batch['context']}")
        print(f"batch['question'] = {batch['question']}")
        print(f"batch['answer_text'] = {batch['answer_text']}")
        print(f"batch['answer_start_idx'] = {batch['answer_start_idx']}")
        print(f"batch['answer_word_start_idx'] = {batch['answer_word_start_idx']}")
        print(f"batch['answer_word_end_idx'] = {batch['answer_word_end_idx']}")
        print(f"batch['input_ids'] = {batch['input_ids']}")
        print(f"batch['words_lengths'] = {batch['words_lengths']}")
        print(f"batch['start_idx'] = {batch['start_idx']}")
        print(f"batch['end_idx'] = {batch['end_idx']}")
        print(f"batch['valid'] = {batch['valid']}")
        break
        # print(batch['tgt_words_ids'].shape)