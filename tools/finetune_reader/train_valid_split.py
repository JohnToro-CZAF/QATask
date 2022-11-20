import os
import datasets
from datasets import concatenate_datasets


def assert_sample(sample):
    assert sample['context'][sample['answer_start_idx']: sample['answer_start_idx'] + len(sample['answer_text'])] == \
           sample['answer_text'], sample
    assert len(sample['context']) > 0
    assert len(sample['question']) > 0
    return True


def format_sample(sample):
    context_prev = sample['context'][:sample['answer_start_idx']].split()
    sample['answer_word_start_idx'] = len(context_prev)
    sample['answer_word_end_idx'] = len(context_prev) + len(sample['answer_text'].split()) - 1
    return sample


def data_split(cfg):
    if not debug_mode and os.path.isfile(cfg.train_path):
        print('%s already exists! Not overwriting.' % cfg.train_path)
        return
    
    train_set = []
    valid_set = []
    mrc_path = cfg.mrc_path
    dataset = datasets.load_dataset('json', data_files=[mrc_path])['train']
    dataset.filter(assert_sample)
    dataset = dataset.map(format_sample)

    all_data = dataset.train_test_split(test_size=0.1)
    train = all_data['train']
    valid = all_data['test']
    train_set.append(train)
    valid_set.append(valid)

    train_dataset = concatenate_datasets(train_set)
    valid_dataset = concatenate_datasets(valid_set)

    train_dataset.save_to_disk(cfg.train_path)
    valid_dataset.save_to_disk(cfg.valid_path)

    print("Train: {} samples".format(len(train_dataset)))
    print("Valid: {} samples".format(len(valid_dataset)))


if __name__ == "__main__":
    # debugging purpose
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrc-path", type=str, default='datasets/data_for_finetuning/mrc_format_file.jsonl')
    parser.add_argument("--train-path", type=str, default='datasets/data_for_finetuning/train.dataset')
    parser.add_argument("--valid-path", type=str, default='datasets/data_for_finetuning/valid.dataset')
    args = parser.parse_args()
    
    data_split(args)