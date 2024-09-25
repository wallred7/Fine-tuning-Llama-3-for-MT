import os
import json
import torch
from torch.nn.parallel import DataParallel
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from transformers import EarlyStoppingCallback
from peft import (LoraConfig, get_peft_model,
                  prepare_model_for_kbit_training)
from trl import SFTTrainer
from config.config import settings
from datetime import datetime



def train(train_file, eval_file, target_language, run_name,
          learning_rate, num_train_epochs, lr_scheduler_type,
          per_device_train_batch_size, lora_dropout, lora_rank):
            
    source_train_file = f'{target_language}_train_dataset.{settings.source_lang_abrv}'
    source_eval_file = f'{target_language}_eval_dataset.{settings.source_lang_abrv}'
    source_train_path = os.path.join(settings.training_data_path, source_train_file)
    source_eval_path = os.path.join(settings.training_data_path, source_eval_file)


    target_train_sentences = load_dataset('text', data_files={'train': train_file}, split='train')
    source_train_sentences = load_dataset('text', data_files={'train': source_train_path}, split='train')
    target_eval_sentences = load_dataset('text', data_files={'validation': eval_file}, split='validation')
    source_eval_sentences = load_dataset('text', data_files={'validation': source_eval_path}, split='validation')

    source_lang = settings.source_language

    prompts = create_training_prompt(source_lang, target_language, source_train_sentences, target_train_sentences)
    eval_prompts = create_training_prompt(source_lang, target_language, source_eval_sentences, target_eval_sentences)

    print(f"Number of prompts: {len(prompts)}")
    n = settings.sample_number
    for prompt in prompts[n:n+5]:
        print(prompt + "\n\n")

    model = load_model(settings.model_path)
    tokenizer = load_tokenizer(settings.model_path)
    model, tokenizer = add_special_tokens(model, tokenizer)
    print("Tokenizer and model loaded successfully.")

    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": prompts[:num_train_records]}),
        "validation": Dataset.from_dict({"text": eval_prompts})
    })

    print(dataset)

    adapter_path = os.path.join(settings.adapter_path, run_name)
    # os.makedirs(adapter_path, exist_ok=True)

    trainer = train_model(
        model, tokenizer, dataset, adapter_path,
        learning_rate, num_train_epochs, lr_scheduler_type,
        per_device_train_batch_size, lora_dropout, lora_rank
    )
    log_training(trainer, adapter_path)

    return adapter_path

def create_training_prompt(source_lang, target_language, new_sources, new_targets, llama_format=True):
    prompts = []
    llama_prompt_format = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for translation from {} to {}. You MUST answer with the following JSON scheme: {{"translation": "string"}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|end_of_text|>'''
    print(llama_prompt_format)
    for new_src, new_trg in zip(new_sources, new_targets):
        source_segment = source_lang + ": " + new_src
        target_segment = f'{{"translation": "{new_trg}"}}'
        if llama_format:
            segment = llama_prompt_format.format(source_lang, target_language, source_segment, target_segment)
        prompts.append(segment)
    return prompts

def prepare_dataset(prompts, eval_prompts, num_train_records):
    return DatasetDict({
        "train": Dataset.from_dict({"text": prompts[:num_train_records]}),
        "validation": Dataset.from_dict({"text": eval_prompts})
    })

def load_model(model_name, tokenizer):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False
    )

    return model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              add_bos_token=True,
                                              add_eos_token=False)
    return tokenizer

def add_special_tokens(model, tokenizer):
    special_tokens_dict = {}
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = '<s>'
    stopping_criteria = tokenizer.convert_ids_to_tokens(tokenizer.encode(settings.end_token))

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def train_model(model, tokenizer, dataset, output_directory,
                learning_rate, num_train_epochs, lr_scheduler_type,
                per_device_train_batch_size, lora_dropout, lora_rank):
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=lora_dropout,  # Updated
        r=lora_rank,                # Updated
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_directory,
        num_train_epochs=num_train_epochs,                    # Updated
        per_device_train_batch_size=per_device_train_batch_size,  # Updated
        per_device_eval_batch_size=per_device_train_batch_size,
        warmup_ratio=0.01,                                    # Use warmup ratio
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",                          # Updated
        learning_rate=learning_rate,                          # Updated
        bf16=True,
        lr_scheduler_type=lr_scheduler_type,                  # Updated
        load_best_model_at_end=True,                          # For early stopping
        metric_for_best_model="eval_loss",                    # Or we can use COMET which is returned by the obkective function in the Runner.py
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        dataset_text_field="text",
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],  # Early stopping
    )

    trainer.train()

    # Print model save path
    print(f"Model saved to: {output_directory}")    

    return trainer 

def log_training(trainer, adapter_path):

    # Collect logs from the training process
    logs = trainer.state.log_history

    # Save detailed logs
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    detailed_logs_path = os.path.join(adapter_path, f"detailed_logs_{date_time}.json")
    with open(detailed_logs_path, "w") as log_file:
        json.dump(logs, log_file, indent=2)

    # Generate and print a report
    report = {
        "output_directory": adapter_path,
        "total_steps": trainer.state.global_step,
        "final_train_loss": logs[-1].get("loss", "N/A"),
        "final_eval_loss": logs[-1].get("eval_loss", "N/A"),
        "epochs_completed": logs[-1].get("epoch", "N/A"),
    }
    report_path = os.path.join(adapter_path, f"experiment_report_{date_time}.json")
    with open(report_path, "w") as report_file:
        json.dump(report, report_file, indent=2)

    print("Experiment Report:")
    print(json.dumps(report, indent=2))
