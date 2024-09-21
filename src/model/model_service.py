import os
import subprocess
import glob

import torch
from torch.nn.parallel import DataParallel

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import ctranslate2

from peft import PeftModel, PeftConfig
from config.config import settings



def prepare_inference_model(adapter_path, fine_tuned_path):
    merged_model = _load_peft_model(adapter_path)
    tokenizer = _load_peft_tokenizer(adapter_path)
    _save_peft_model(merged_model, fine_tuned_path)
    _save_peft_tokenizer(tokenizer, fine_tuned_path)

    ct2_directory = _convert_model_to_ct2(fine_tuned_path, settings.ctranslate2_path)
    generator, tokenizer = _load_ct2_model(ct2_directory)
    return generator, tokenizer

def _load_peft_model(adapter_path):
    # Since the peft lora weights are saved seperately from the model we load them and save the merged model. 
    peftconfig = PeftConfig.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(peftconfig.base_model_name_or_path, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = model.merge_and_unload()
    print("Peft model loaded")
    return merged_model

def _load_peft_tokenizer(adapter_path):
    peftconfig = PeftConfig.from_pretrained(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(peftconfig.base_model_name_or_path)
    print('Peft tokenizer loaded')
    return tokenizer

def _save_peft_model(merged_model, fine_tuned_path):
    merged_model.save_pretrained(fine_tuned_path)
    print(f"Model saved to {fine_tuned_path}")

def _save_peft_tokenizer(tokenizer, fine_tuned_path):
    tokenizer.save_pretrained(fine_tuned_path)
    print(f"Tokenizer saved to {fine_tuned_path}")

def get_latest_checkpoint(checkpoint_dir): # This may become important later
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*'))
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1])).split('-')[-1]
    return latest_checkpoint

def _convert_model_to_ct2(fine_tuned_path, ctranslate2_path):
    # converts the model to CTranslate2 to enable batch processing run via command 
    # ct2_save_directory = os.path.join(ctranslate2_path, f"{settings.model_name}") #TODO make sure the path is correct
    run_name = os.path.basename(fine_tuned_path)
    ct2_save_directory = os.path.join(ctranslate2_path, run_name)
    os.makedirs(ct2_save_directory, exist_ok=True)
    print(f"Converting model to CTranslate2 format and saving to {ct2_save_directory}...")
    subprocess.run([
        "ct2-transformers-converter",
        "--model", fine_tuned_path,
        "--quantization", "int8",
        "--output_dir", ct2_save_directory,
        "--force"
    ], check=True, text=True)
    print("Conversion to CTranslate2 format completed.")
    return ct2_save_directory


def _load_ct2_model(ctranslate2_path):
    generator = ctranslate2.Generator(ctranslate2_path, device="cuda", compute_type="int8")
    tokenizer = transformers.AutoTokenizer.from_pretrained(settings.model_name)
    print("Model and tokenizer loaded.")
    print(f"Model: {ctranslate2_path}")
    print(f"Tokenizer: {settings.model_name}")
    return generator, tokenizer

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
