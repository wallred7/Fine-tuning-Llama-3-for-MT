import os
import torch
from torch.nn.parallel import DataParallel

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import ctranslate2

from peft import PeftModel, PeftConfig
from model.collection import load_translations_file
from model.model_service import create_prompt, prepare_inference_model
from config.config import settings


def run_inference(target_language, adapter_path, run_name):
    
    target_languages = {
        'pt-br': 'Brazilian Portuguese',
        'cs': 'Czech',
        'de': 'German',
        'fi': 'Finnish',
        'ko': 'Korean'
    }
    
    target_lang = target_languages[target_language]
    source_sentences = load_translations_file(settings.source_data_path)
    
    prompts = create_prompt(settings.source_language, target_lang, source_sentences, llama_format=True) #TODO double check the prompt format
    n = settings.sample_number
    print(f'Number of prompts: {len(prompts)}')
    print(f'Example prompt {n}:\n{prompts[n]}\n')
    
    fine_tuned_path = os.path.join(settings.fine_tuned_path, run_name)

    generator, tokenizer = prepare_inference_model(adapter_path, fine_tuned_path)
    
    lengths = [len(new_src.split()) for new_src in source_sentences]
    length_multiplier = settings.length_multiplier
    max_len = max(lengths) * length_multiplier

    translations = translate_batch(prompts, tokenizer, generator, max_len, "}assistant", topk=settings.topk)
    
    print(f"Number of translations: {len(translations)}")
    for i, (source, translation) in enumerate(zip(source_sentences[n:n+5], translations[n:n+5]), 1):
        print(f"Source {i}: {source}")
        print(f"Translation {i}: {translation} (Length: {len(translation)})")
    
    cleaned_translations = clean_translation(translations)
    output_path = os.path.join(settings.output_data_path, run_name)
    translations_file_name = save_translations(cleaned_translations, output_path, target_language, run_name)
    return translations_file_name


def translate_batch(prompts, tokenizer, generator, max_length, end_token, topk=1):
    print("Starting batch translation...")
    tokenized_inputs = tokenizer(prompts)
    input_ids_batch = tokenized_inputs['input_ids']
    tokens_batch = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids_batch]

    results = generator.generate_batch(tokens_batch,
                                       sampling_topk=topk,
                                       max_length=max_length,
                                       min_length=1,
                                       include_prompt_in_result=False,
                                       end_token=end_token,
                                       batch_type="tokens",
                                       max_batch_size=8096)
    sequences_ids = []
    for result in results:
        sequence = result.sequences_ids[0]
        if end_token in sequence:
            end_index = sequence.index(end_token)
            sequence = sequence[:end_index]  # Truncate just before the stop token
        sequences_ids.append(sequence)
    translations = tokenizer.batch_decode(sequences_ids, skip_special_tokens=True)
    print("Batch translation completed.")
    return translations

def remove_tags_after_bracket(text):
    # Specific cleaning functionality occasionally required due to overgeneration
    bracket_index = text.find('}')
    if (bracket_index != -1):
        cleaned_text = text[:bracket_index]
    else:
        cleaned_text = text
    return cleaned_text

def clean_translation(translation):
    no_tags_translation = remove_tags_after_bracket(translation)
    no_json_translation = no_tags_translation.replace('{"translation": "', '').rstrip('"}')
    cleaned_translation = no_json_translation.replace('\n', ' ')
    return cleaned_translation

def save_translations(translations, output_dir, target1, run_name):
    # cleans and saves the translations 
    translations_file_name = os.path.join(output_dir, f"{target1}_{run_name}_translations.txt")
    with open(translations_file_name, "w+", encoding="utf-8") as output:
        for translation in translations:
            cleaned_translation = clean_translation(translation)
            output.write(cleaned_translation + "\n")
    print(f"Translations saved to {translations_file_name}")
    return translations_file_name
