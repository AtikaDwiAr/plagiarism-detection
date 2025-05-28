import os
import nltk
import re
import ssl
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Setup NLTK Auto-Installer
def setup_nltk():
    # Mengatasi masalah NLTK resource di semua device
    try:
        # Bypass SSL verification
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Set universal path
        nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)

        # Download required resources
        resources = {
            'tokenizers/punkt': 'punkt',
            'corpora/stopwords': 'stopwords',
            'corpora/wordnet': 'wordnet',
            'tokenizers/punkt_tab': 'punkt_tab'
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

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Text Preprocessing functions
def text_lowercase(text):
    return text.lower()

def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

def convert_number(text):
    return re.sub(r'\d+', '', text)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_whitespace(text):
    return " ".join(text.split())

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return ' '.join(stems)

def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    return ' '.join(lemmas)

# Main Preprocessing Function
def preprocess_text(text):
    """Fungsi preprocessing utama dengan dictionary-friendly output"""
    try:
        text = text_lowercase(text)
        text = convert_number(text)
        text = remove_numbers(text) 
        text = remove_punctuation(text)
        text = remove_whitespace(text)
        text = remove_stopwords(text)
        text = stem_words(text)
        text = lemmatize_word(text)
        return text
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text  # Return original text if error occurs

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_count = 0
    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"preprocessed_{filename}")
        
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            processed_text = preprocess_text(raw_text)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Processed {processed_count} files from {input_dir} to {output_dir}")

# Main Execution
if __name__ == "__main__":
    # Initialize NLTK
    setup_nltk()

    # Path configuration (sesuai struktur PAN-PC-11)
    SOURCE_DIR = 'pan_dataset/external-detection-corpus/source-document/part1'
    SUSPICIOUS_DIR = 'pan_dataset/external-detection-corpus/suspicious-document/part1'
    
    OUTPUT_SOURCE = 'preprocessed_data/source'
    OUTPUT_SUSPICIOUS = 'preprocessed_data/suspicious'
    
    # Process documents
    print("\nProcessing source documents...")
    process_directory(SOURCE_DIR, OUTPUT_SOURCE)
    
    print("\nProcessing suspicious documents...")
    process_directory(SUSPICIOUS_DIR, OUTPUT_SUSPICIOUS)
    
    print("\nPreprocessing completed successfully!")
