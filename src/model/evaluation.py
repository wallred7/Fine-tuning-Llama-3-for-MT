import os
import sacrebleu
import pandas as pd
from datetime import datetime
from comet import download_model, load_from_checkpoint
from model.collection import load_test_datasets, load_translations
from config.config import settings


def evaluate_output(output_file, lang, run_name):
    # load translations & print samples
    source_sentences = load_translations(settings.source_data_path)
    reference_sentences = load_translations(settings.reference_data_path)
    output_sentences = load_translations(output_file)
    print_sample_sentences(source_sentences, reference_sentences, output_sentences)

    check_missing_translations(source_sentences, reference_sentences)
    
    # evaluate 
    bleu, chrf, ter, comet = calculate_metrics(source_sentences, reference_sentences, output_sentences)

    # Save and append results to file 
    df = pd.DataFrame({"Language": lang, "BLEU": [bleu], "chrF++": [chrf], "TER": [ter], "COMET": [comet]}) #todo switch to pandas
    print(df.head())

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    df.to_csv(f"{settings.metric_results_path}/{run_name}/result_metrics_{lang}_{run_name}_{date_time}.csv")

def print_sample_sentences(source_sentences, reference_sentences, output_sentences, n=settings.sample_number):
    print(source_sentences[n])
    print(reference_sentences[n])
    print(output_sentences[n])


def check_missing_translations(source_sentences, reference_sentences, translations=-1):
    count = 0
    for idx, line in enumerate(translations):
        if len(line.strip()) == 0:
            count += 1
            print(idx, source_sentences[idx].strip(), reference_sentences[idx].strip(), sep="\n", end="\n\n")
    print("Missing translations:", count)


def calculate_comet(source_sentences, reference_sentences, output_sentences):
    print('src len: ', len(source_sentences), '\ntarg len: ', len(output_sentences), '\nref len: ', len(reference_sentences))
    df = pd.DataFrame({"src": source_sentences, "mt": output_sentences, "ref": reference_sentences})
    data = df.to_dict('records')

    if not os.path.exists(settings.comet_path):
        model_path = download_model("wmt20-comet-da", saving_directory=settings.comet_path)

    model = load_from_checkpoint(settings.comet_path)

    seg_scores, sys_score = model.predict(data, batch_size=128, gpus=1).values()
    comet = round(sys_score * 100, 2)
    print("COMET:", comet)

    return comet


def calculate_metrics(source_sentences, reference_sentences, output_sentences):
    bleu = sacrebleu.corpus_bleu(output_sentences, [reference_sentences])
    bleu = round(bleu.score, 2)
    print("BLEU:", bleu)

    chrf = sacrebleu.corpus_chrf(output_sentences, [reference_sentences], word_order=2)
    chrf = round(chrf.score, 2)
    print("chrF++:", chrf)

    metric = sacrebleu.metrics.TER()
    ter = metric.corpus_score(output_sentences, [reference_sentences])
    ter = round(ter.score, 2)
    print("TER:", ter)

    comet = calculate_comet(source_sentences, reference_sentences, output_sentences)

    return bleu, chrf, ter, comet
