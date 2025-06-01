import os
import nltk
import re
import ssl
import string
import csv
import pandas as pd

from langdetect import detect, LangDetectException
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Setup NLTK Auto-Installer
def setup_nltk():
    try:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)

        resources = {
            'tokenizers/punkt': 'punkt',
            'corpora/stopwords': 'stopwords',
            'corpora/wordnet': 'wordnet',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
        }

        for path, package in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading NLTK resource: {package}")
                nltk.download(package, download_dir=nltk_data_path)

    except Exception as e:
        print(f"NLTK setup error: {e}")
        exit(1)

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# --- Preprocessing Functions ---
def text_lowercase(text):
    return text.lower()

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_whitespace(text):
    return " ".join(text.split())

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_word(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
    ]
    return ' '.join(lemmatized)

# Fungsi deteksi bahasa
def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

# Pipeline utama
def preprocess_text(text):
    try:
        text = text_lowercase(text)
        text = remove_numbers(text)
        text = remove_punctuation(text)
        text = remove_whitespace(text)
        text = remove_stopwords(text)
        text = lemmatize_word(text)
        return text
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text

# Proses direktori
def process_directory(input_dir, output_dir, csv_output_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = []
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"preprocessed_{filename}")

        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()

            if not is_english(raw_text):
                print(f"Skipped non-English file: {filename}")
                skipped_count += 1
                continue

            processed_text = preprocess_text(raw_text)

            # Simpan hasil ke file .txt
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)

            # Simpan hasil ke CSV list
            rows.append([filename, processed_text])
            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Simpan ke CSV
    with open(csv_output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'preprocessed_text'])
        writer.writerows(rows)

    print(f"\nProcessed {processed_count} English files from {input_dir} to {output_dir}")
    print(f"Skipped {skipped_count} non-English files.")

# Main
if __name__ == "__main__":
    setup_nltk()

    SOURCE_DIR = 'pan_dataset/external-detection-corpus/source-document/part1'
    SUSPICIOUS_DIR = 'pan_dataset/external-detection-corpus/suspicious-document/part1'

    OUTPUT_SOURCE = 'preprocessed_data/source'
    OUTPUT_SUSPICIOUS = 'preprocessed_data/suspicious'

    CSV_SOURCE = 'preprocessed_data/preprocessed_source.csv'
    CSV_SUSPICIOUS = 'preprocessed_data/preprocessed_suspicious.csv'

    print("\nProcessing source documents...")
    process_directory(SOURCE_DIR, OUTPUT_SOURCE, CSV_SOURCE)

    print("\nProcessing suspicious documents...")
    process_directory(SUSPICIOUS_DIR, OUTPUT_SUSPICIOUS, CSV_SUSPICIOUS)

    print("\nPreprocessing completed successfully!")
