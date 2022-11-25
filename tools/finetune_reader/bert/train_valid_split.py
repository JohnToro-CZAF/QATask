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
    class Config:
        def __init__(self) -> None:
            self.mrc_path = os.path.join(os.getcwd(), 'qatask/database/datasets/data_for_finetuning/mrc_format_file.jsonl')
            self.train_path = os.path.join(os.getcwd(), 'qatask/database/datasets/data_for_finetuning/train.dataset')
            self.valid_path = os.path.join(os.getcwd(), 'qatask/database/datasets/data_for_finetuning/valid.dataset')
    
    config = Config()
    data_split(config)