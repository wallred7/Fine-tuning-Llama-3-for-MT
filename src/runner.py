import os

from model.training import train
from model.inference import run_inference
from model.evaluation import evaluate_output
from model.model_service import get_latest_checkpoint
from config.config import settings

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
