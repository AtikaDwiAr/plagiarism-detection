import os
import re
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
            'corpora/stopwords': 'stopwords'
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

# Text Preprocessing
def preprocess_text(text):
    """Fungsi preprocessing utama"""
    # Case folding
    text = text.lower()
    
    # Remove special chars/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization with fallback
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()  # Basic fallback
    
    # Stopword removal
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except:
        pass
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

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