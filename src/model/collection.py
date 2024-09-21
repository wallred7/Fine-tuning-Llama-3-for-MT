import pandas as pd
from datasets import load_dataset


def load_data(train_file, eval_file):
    train_data = load_dataset('csv', data_files={'train': train_file}, split='train')
    eval_data = load_dataset('csv', data_files={'eval': eval_file}, split='eval')
    return train_data, eval_data


def load_translations_file(translations_file_path):
    with open(translations_file_path, 'r', encoding='utf-8') as f:
        translations = [line.strip() for line in f]
    return translations
