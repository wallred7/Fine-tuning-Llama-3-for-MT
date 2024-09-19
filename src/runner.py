from model.training import train
from model.inference import run_inference
from model.evaluation import evaluate_output
from config.config import settings

for i in settings.lang_abrv:
    fine_tuned_model = train() # args: train_file, eval_file, train_lang (i), num_train_records
    translations_files = run_inference(i, fine_tuned_model)
    evaluate_output(translations_files)

