import os
import sys
import subprocess
import glob

import torch
from torch.nn.parallel import DataParallel

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import ctranslate2

from peft import PeftModel, PeftConfig
from model.collection import load_translations_file, create_prompt
from config.config import settings


def run_inference(target_language, fine_tune_model):
    
    target_languages = {
        'pt-br': 'Brazilian Portuguese',
        'cs': 'Czech',
        'de': 'German',
        'fi': 'Finnish',
        'ko': 'Korean'
    }
    
    source_lang = "English"
    target_lang = target_languages[target_language]
    # size = settings.size.replace('\'','')
    # dataset = 'test'
    # ft_model = f'llama-3-8B-{target_lang}-{size}'.replace(' ','_')

    source_sentences = load_translations_file(settings.source_data_path)
    
    prompts = create_prompt(source_lang, target_lang, source_sentences, llama_format=True) #TODO double check the prompt format
    print(f'Number of prompts: {len(prompts)}')
    for i, prompt in enumerate(prompts[:10], 1):
        print(f'Example prompt {i}:\n{prompt}\n')

    latest_checkpoint = get_latest_checkpoint(fine_tune_model)
    
    fine_tune_model_path = os.path.join(fine_tune_model, f'checkpoint-{latest_checkpoint}')
    model, tokenizer = load_peft_model(settings.adapter_path, settings.fine_tuned_path)

    generator, tokenizer = convert_model_to_ct2(model, tokenizer, settings.fine_tuned_path, settings.ctranslate_path, fine_tune_model)
    
    lengths = [len(new_src.split()) for new_src in source_sentences]
    length_multiplier = 2
    max_len = max(lengths) * length_multiplier

    translations = translate_batch(prompts, tokenizer, generator, max_len, "}assistant", topk=1)
    
    print(f"Number of translations: {len(translations)}")
    for i, (source, translation) in enumerate(zip(source_sentences[:10], translations[:10]), 1):
        print(f"Source {i}: {source}")
        print(f"Translation {i}: {translation} (Length: {len(translation)})")
    
    translations_file_name = save_translations(translations, settings.output_data_path, target_language, fine_tune_model)
    return translations_file_name


def load_peft_model(adapter_path, final_model_path):
    # Since the peft lora weights are saved seperately from the model we load them and save the merged model. 
    peftconfig = PeftConfig.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(peftconfig.base_model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(peftconfig.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Peft model loaded")

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model and tokenizer saved to {final_model_path}")

    return merged_model, tokenizer

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*'))
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1])).split('-')[-1]

    return latest_checkpoint

def convert_model_to_ct2(model, tokenizer, save_dir, ct2_model_dir):
    # converts the model to CTranslate2 to enable batch processing run via command 
    ct2_save_directory = os.path.join(ct2_model_dir, f"{settings.model_name}")
    os.makedirs(ct2_save_directory, exist_ok=True)
    print(f"Converting model to CTranslate2 format and saving to {ct2_save_directory}...")
    subprocess.run([
        "ct2-transformers-converter",
        "--model", save_dir,
        "--quantization", "int8",
        "--output_dir", ct2_save_directory,
        "--force"
    ], check=True, text=True)
    print("Conversion to CTranslate2 format completed.")

    generator = ctranslate2.Generator(ct2_save_directory, device="cuda", compute_type="int8")
    tokenizer = transformers.AutoTokenizer.from_pretrained(settings.model_name)
    print("Model and tokenizer loaded.")
    print(f"Model: {ct2_save_directory}")
    print(f"Tokenizer: {settings.model_name}")
    return generator, tokenizer

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

def save_translations(translations, output_dir, target1, ft_model):
    # cleans and saves the translations 
    translations_file_name = os.path.join(output_dir, f"final_{target1}_{ft_model}_translations.txt")
    with open(translations_file_name, "w+", encoding="utf-8") as output:
        for translation in translations:
            translation = remove_tags_after_bracket(translation)
            actual_translation = translation.replace('{"translation": "', '').rstrip('"}')
            cleaned_translation = actual_translation.replace('\n', ' ')
            output.write(cleaned_translation + "\n")
    print(f"Translations saved to {translations_file_name}")
    return translations_file_name
