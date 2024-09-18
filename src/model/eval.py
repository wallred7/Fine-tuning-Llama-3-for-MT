import os
import sacrebleu
import pandas as pd
from comet import download_model, load_from_checkpoint
import polars as pl
import argparse
import ast

def set_data_path(data_path):
#    home = os.path.expanduser("~")
#    data_path = os.path.join(home, data_path)
    os.chdir(data_path)
    return data_path

def load_test_datasets(data_path, source_test_file, target_test_file):
    with open(os.path.join(data_path, source_test_file), encoding="utf-8") as source, open(os.path.join(data_path, target_test_file), encoding="utf-8") as target:
        source_sentences = [sent.strip() for sent in source.readlines()]
        target_sentences = [sent.strip() for sent in target.readlines()]

    return source_sentences, target_sentences

def print_sample_sentences(source_sentences, target_sentences, n=57):
    print(source_sentences[n])
    print(target_sentences[n])

def load_translations(translations_file_path):
    with open(translations_file_path, encoding="utf-8") as translated:
        translations = [sent.strip() for sent in translated.readlines()]

    print(translations_file_path, "\n")
    print(*translations[0:5], sep="\n")
    return translations

def check_missing_translations(source_sentences, target_sentences, translations):
    count = 0
    for idx, line in enumerate(translations):
        if len(line.strip()) == 0:
            count += 1
            print(idx, source_sentences[idx].strip(), target_sentences[idx].strip(), sep="\n", end="\n\n")
    print("Missing translations:", count)

def calculate_comet(data_path, source_sentences, translations, references, file=None):
    print('src len: ', len(source_sentences), '\ntarg len: ', len(translations), '\nref len: ', len(references))
    df = pd.DataFrame({"src": source_sentences, "mt": translations, "ref": references})
    data = df.to_dict('records')

    #model_dir = os.path.join(data_path, "models/wmt22-comet-da/checkpoints")
    model_dir = os.path.join(data_path, "models/wmt20-comet-da/checkpoints")
    model_path = os.path.join(model_dir, "model.ckpt")

    if not os.path.exists(model_path):
        #model_path = download_model("Unbabel/wmt22-comet-da", saving_directory=model_dir)
        model_path = download_model("wmt20-comet-da", saving_directory=model_dir)

    model = load_from_checkpoint(model_path)

    seg_scores, sys_score = model.predict(data, batch_size=128, gpus=1).values()
    comet = round(sys_score * 100, 2)
    print("COMET:", comet)

    # Get the 10 best and worst scoring sentences
    sorted_indices = sorted(range(len(seg_scores)), key=lambda i: seg_scores[i])
    worst_indices = sorted_indices[:10]
    best_indices = sorted_indices[-10:][::-1]

    print("\n10 Worst Scoring Sentences:")
    for idx in worst_indices:
        print(f"Position: {idx}, Score: {seg_scores[idx]:.2f}, Source: {source_sentences[idx]}, Reference: {references[idx]}, Translation: {translations[idx]}")

    print("\n10 Best Scoring Sentences:")
    for idx in best_indices:
        print(f"Position: {idx}, Score: {seg_scores[idx]:.2f}, Source: {source_sentences[idx]}, Reference: {references[idx]}, Translation: {translations[idx]}")

    #1000's best/worst translations from pt-br (set manually) 
    if file:
        bw = [416, 1223, 880, 1340, 1094, 492, 1000, 983, 780, 30, 681, 1054, 1032, 775, 1057, 1064, 1031, 1028, 1091]

        output_file = '/home/wallred/practicum/bp_quality_check.csv'
        with open(output_file, 'a') as f:
            for idx in bw:
                score = round(seg_scores[idx] * 100, 2)
                output = f"File: {file}, Position: {idx}, Score: {score}, Source: {source_sentences[idx]}, Reference: {references[idx]}, Translation: {translations[idx]}"
                f.write(f'{file},{idx},{score},{source_sentences[idx]},{references[idx]},{translations[idx]}')
                f.write('\n')
                print(output)

    return comet

def calculate_metrics(data_path, source_sentences, translations, references, file=None):
    bleu = sacrebleu.corpus_bleu(translations, [references])
    bleu = round(bleu.score, 2)
    print("BLEU:", bleu)

    chrf = sacrebleu.corpus_chrf(translations, [references], word_order=2)
    chrf = round(chrf.score, 2)
    print("chrF++:", chrf)

    metric = sacrebleu.metrics.TER()
    ter = metric.corpus_score(translations, [references])
    ter = round(ter.score, 2)
    print("TER:", ter)

    comet = calculate_comet(data_path, source_sentences, translations, references, file)

    return bleu, chrf, ter, comet

def append_or_create_csv(df, csv_file, lang, size):
    try:
        existing_df = pl.scan_csv(csv_file)
        existing_df = existing_df.collect()

        # Convert df data types to match existing_df
        for col in existing_df.columns:
            if col in df.columns:
                existing_dtype = existing_df[col].dtype
                df = df.with_columns(pl.col(col).cast(existing_dtype))

        concatenated_df = pl.concat([existing_df, df], rechunk=False)
        concatenated_df.write_csv(csv_file)
    except Exception as e:
        print(f'Read CSV Error: {e}')
        df.write_csv(f'{lang}_{size}_{csv_file}')

def main(data_path, source_test_file, target_test_file, translations_file_name):

    # arguments 
    print(data_path, source_test_file, target_test_file, translations_file_name)
    
    # path
    data_path = set_data_path(data_path)
    
    # load translations & print samples
    source_sentences, target_sentences = load_test_datasets(data_path, source_test_file, target_test_file)
    print_sample_sentences(source_sentences, target_sentences)
    output_translations = os.path.join(data_path, translations_file_name)
    print('translations path: ', output_translations)
    translations = load_translations(output_translations)
    
    # evaluate 
    end = -1 # allows user to set a smaller number of translations to evaluate
    check_missing_translations(source_sentences, target_sentences, translations[:end])

    references = target_sentences[:end]
    bleu, chrf, ter, comet = calculate_metrics(data_path, source_sentences[:end], translations[:end], references, translations_file_name)

    lang = str(target_test_file).rsplit('.', 1)[1]
    try:
        size = int(''.join(filter(str.isdigit, translations_file_name))[2:]) # extracts the dataset size from the file name 
    except:
        size = '000' # an error indicates the baseline or a different file format 

    # Save and append results to file 
    df = pl.DataFrame({"Language": lang, "Size": size, "BLEU": [bleu], "chrF++": [chrf], "TER": [ter], "COMET": [comet]})
    print(df.head())
    append_or_create_csv(df, 'results.csv', lang, size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate translation models.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--source_test_file", type=str, required=True, help="Name of the source test file")
    parser.add_argument("--target_test_file", type=str, required=True, help="Name of the target test file")
    parser.add_argument("--translations_file_name", type=str, required=True, help="Name of the translations file")
    args = parser.parse_args()
    main(args.data_path, args.source_test_file, args.target_test_file, args.translations_file_name)