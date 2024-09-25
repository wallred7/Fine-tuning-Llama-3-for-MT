import os

from model.training import train
from model.inference import run_inference
from model.evaluation import evaluate_output
from model.model_service import get_latest_checkpoint
from config.config import settings

def objective(trial):
    lang = settings.lang_abrv[0]  # You can loop over languages as needed

    target_languages = {
        'pt-br': 'Brazilian Portuguese',
        'cs': 'Czech',
        'de': 'German',
        'fi': 'Finnish',
        'ko': 'Korean'
    }

    target_language = target_languages[lang]

    run_name = f"{lang}_{settings.model_name.replace('.','_')}_trial_{trial.number}"
    train_file = os.path.join(settings.training_data_path, f'train_dataset.{lang}')
    eval_file = os.path.join(settings.training_data_path, f'eval_dataset.{lang}')

    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)  # Reduced range
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 5)        # Increased epochs
    lr_scheduler_type = trial.suggest_categorical('lr_scheduler_type', ['linear', 'cosine'])
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [32, 64, 128])
    lora_dropout = trial.suggest_uniform('lora_dropout', 0.1, 0.3)
    lora_rank = trial.suggest_categorical('lora_rank', [16, 32])

    # Train the model
    adapter_path = train(
        train_file, eval_file, lang, run_name,
        learning_rate, num_train_epochs, lr_scheduler_type,
        per_device_train_batch_size, lora_dropout, lora_rank
    )

    # Inference and evaluation
    translations_files = run_inference(lang, adapter_path, run_name)
    metrics = evaluate_output(translations_files, lang, run_name)

    # Return the metric to optimize (e.g., COMET score)
    return metrics["COMET"]
    
def main():
    for lang in settings.lang_abrv:
        target_languages = {
                'pt-br': 'Brazilian Portuguese',
                'cs': 'Czech',
                'de': 'German',
                'fi': 'Finnish',
                'ko': 'Korean'
            }

        target_language = target_languages[lang]

        run_name = f"{lang}_{settings.model_name.replace('.','_')}"
        train_file = os.path.join(settings.training_data_path, f'train_dataset.{lang}')
        eval_file = os.path.join(settings.training_data_path, f'eval_dataset.{lang}') # DEV set

        fine_tuned_model = train(train_file, eval_file, lang, run_name)
        fine_tuned_model_ckpt = get_latest_checkpoint(fine_tuned_model)
        translations_files = run_inference(lang, fine_tuned_model_ckpt, run_name)
        evaluate_output(translations_files, lang, run_name)

if __name__=="__main__":
    main() 
