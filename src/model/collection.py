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

def create_prompt(source_lang, target_lang, sources, targets=None, llama_format=True): #TODO double check that the training and inference prompts are distinct
    prompts = []
    llama_prompt_format = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for translation from {} to {}. You MUST answer with the following JSON scheme: {{"translation": "string"}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|end_of_text|>'''

    for i, source in enumerate(sources):
        source_segment = source_lang + ": " + source
        target_segment = ''
        if targets:
            target_segment = f'{{"translation": "{targets[i]}"}}'
        if llama_format:
            if targets: # training
                segment = llama_prompt_format.format(source_lang, target_lang, source_segment, target_segment)
            else: # inference
                segment = llama_prompt_format.format(source_lang, target_lang, source_segment, '')[:-23] # removes assistant and end of text tags
        prompts.append(segment)
    return prompts