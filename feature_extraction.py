import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch

# Pastikan sudah menjalankan loaddata.py dan preprocessing.py
# agar folder 'preprocessed_data' sudah ada

# Jika ingin mengimpor fungsi preprocess_text langsung, Anda bisa melakukannya:
# from preprocessing import preprocess_text, setup_nltk

def load_preprocessed_documents(directory_path):
    """
    Memuat semua dokumen dari direktori yang telah diproses. Asumsi dokumen adalah file .txt dan sudah diproses.
    """
    documents = []
    filenames = []
    if not os.path.exists(directory_path):
        print(f"Direktori tidak ditemukan: {directory_path}")
        return [], []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                documents.append(f.read())
            filenames.append(filename)
    return documents, filenames

def extract_bow_features(corpus):
    """
    Mengekstrak fitur Bag of Words (BoW) dari korpus teks.
    """
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,1)) # max_features bisa disesuaikan
    bow_matrix = vectorizer.fit_transform(corpus)
    return bow_matrix, vectorizer

def extract_tfidf_features(corpus):
    """
    Mengekstrak fitur TF-IDF dari korpus teks.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2)) # ngram_range bisa disesuaikan
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def train_word2vec_model(sentences):
    """
    Melatih model Word2Vec. 'sentences' adalah list of list of string (list of tokenized sentences). Contoh: [['kata', 'satu'], ['kata', 'dua']]
    """
    # Parameter Word2Vec bisa disesuaikan (vector_size, window, min_count, workers, etc.)
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_document_embedding_w2v(doc_tokens, word2vec_model):
    """
    Mendapatkan embedding dokumen dengan merata-ratakan embedding kata-kata.
    """
    embeddings = []
    for token in doc_tokens:
        if token in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[token])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size) # Mengembalikan vektor nol jika tidak ada kata

def get_bert_embedding(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Mendapatkan embedding dokumen menggunakan model BERT (Sentence-Transformers). Membutuhkan koneksi internet untuk mengunduh model pertama kali.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        print("Pastikan Anda memiliki koneksi internet atau model sudah diunduh.")
        return None

    # Tokenisasi dan embedding batch untuk efisiensi
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Ambil embedding [CLS] token (biasanya digunakan untuk representasi kalimat)
    # Atau rata-ratakan embedding dari semua token (mean pooling)
    sentence_embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
    return sentence_embeddings


if __name__ == "__main__":
    # Jika preprocessing.py dijalankan sebagai skrip utama, maka tidak perlu setup
    # setup_nltk() 

    # Path konfigurasi (sesuai output dari preprocessing.py)
    PREPROCESSED_SOURCE_DIR = 'preprocessed_data/source'
    PREPROCESSED_SUSPICIOUS_DIR = 'preprocessed_data/suspicious'

    print("Memuat dokumen sumber yang sudah diproses...")
    source_docs, source_filenames = load_preprocessed_documents(PREPROCESSED_SOURCE_DIR)
    print(f"Ditemukan {len(source_docs)} dokumen sumber.")

    print("Memuat dokumen mencurigakan yang sudah diproses...")
    suspicious_docs, suspicious_filenames = load_preprocessed_documents(PREPROCESSED_SUSPICIOUS_DIR)
    print(f"Ditemukan {len(suspicious_docs)} dokumen mencurigakan.")

    if not source_docs or not suspicious_docs:
        print("Tidak ada dokumen yang ditemukan. Pastikan preprocessing.py sudah dijalankan dan menghasilkan output.")
    else:
        # --- Bag of Words (BoW) ---
        print("\n--- Mengekstrak fitur Bag of Words (BoW) ---")
        all_docs_for_bow = source_docs + suspicious_docs
        bow_matrix, bow_vectorizer = extract_bow_features(all_docs_for_bow)
        
        bow_source_matrix = bow_matrix[:len(source_docs)]
        bow_suspicious_matrix = bow_matrix[len(source_docs):]
        
        print(f"Bentuk matriks BoW sumber: {bow_source_matrix.shape}")
        print(f"Bentuk matriks BoW mencurigakan: {bow_suspicious_matrix.shape}")

        # --- TF-IDF ---
        print("\n--- Mengekstrak fitur TF-IDF ---")
        all_docs_for_tfidf = source_docs + suspicious_docs
        tfidf_matrix, tfidf_vectorizer = extract_tfidf_features(all_docs_for_tfidf)
        
        tfidf_source_matrix = tfidf_matrix[:len(source_docs)]
        tfidf_suspicious_matrix = tfidf_matrix[len(source_docs):]
        
        print(f"Bentuk matriks TF-IDF sumber: {tfidf_source_matrix.shape}")
        print(f"Bentuk matriks TF-IDF mencurigakan: {tfidf_suspicious_matrix.shape}")

        # --- Word2Vec (Word Embedding) ---
        print("\n--- Mengekstrak fitur Word2Vec Embeddings ---")
        # Tokenisasi dokumen untuk Word2Vec
        tokenized_source_docs = [doc.split() for doc in source_docs]
        tokenized_suspicious_docs = [doc.split() for doc in suspicious_docs]
        all_tokenized_docs = tokenized_source_docs + tokenized_suspicious_docs

        if all_tokenized_docs:
            print("Melatih model Word2Vec...")
            word2vec_model = train_word2vec_model(all_tokenized_docs)

            print("Mendapatkan embedding dokumen sumber Word2Vec...")
            word2vec_source_embeddings = np.array([get_document_embedding_w2v(tokens, word2vec_model) for tokens in tokenized_source_docs])
            print(f"Bentuk embedding Word2Vec sumber: {word2vec_source_embeddings.shape}")

            print("Mendapatkan embedding dokumen mencurigakan Word2Vec...")
            word2vec_suspicious_embeddings = np.array([get_document_embedding_w2v(tokens, word2vec_model) for tokens in tokenized_suspicious_docs])
            print(f"Bentuk embedding Word2Vec mencurigakan: {word2vec_suspicious_embeddings.shape}")
        else:
            print("Tidak ada token untuk melatih Word2Vec.")

        # --- BERT Word2Vec (Menggunakan Sentence-Transformers sebagai representasi BERT yang efisien) ---
        # Catatan: BERT tidak menghasilkan "word2vec" dalam arti tradisional.
        # Biasanya, kita mendapatkan embedding tingkat kalimat/dokumen dari BERT.
        # Di sini, kita akan menggunakan model Sentence-Transformers yang berbasis BERT
        # untuk mendapatkan embedding dokumen yang bagus.
        print("\n--- Mengekstrak fitur BERT Word2Vec (Document Embeddings) ---")
        
        # Pastikan sudah memiliki model BERT yang sesuai.
        # Model 'sentence-transformers/all-MiniLM-L6-v2' adalah pilihan yang ringan dan efektif
        # Pastikan untuk menginstal transformers dan torch (pip install transformers torch sentence-transformers)
        
        # Hanya ambil beberapa dokumen untuk demonstrasi karena BERT bisa intensif komputasi
        sample_size = min(5, len(source_docs), len(suspicious_docs))
        
        if sample_size > 0:
            sample_source_docs_bert = source_docs[:sample_size]
            sample_suspicious_docs_bert = suspicious_docs[:sample_size]

            print(f"Mendapatkan embedding BERT untuk {sample_size} dokumen sumber (sampel)...")
            bert_source_embeddings_sample = get_bert_embedding(sample_source_docs_bert)
            if bert_source_embeddings_sample is not None:
                print(f"Bentuk embedding BERT sumber (sampel): {bert_source_embeddings_sample.shape}")

            print(f"Mendapatkan embedding BERT untuk {sample_size} dokumen mencurigakan (sampel)...")
            bert_suspicious_embeddings_sample = get_bert_embedding(sample_suspicious_docs_bert)
            if bert_suspicious_embeddings_sample is not None:
                print(f"Bentuk embedding BERT mencurigakan (sampel): {bert_suspicious_embeddings_sample.shape}")
        else:
            print("Tidak cukup dokumen untuk demonstrasi BERT.")
        
        print("\nDemonstrasi Cosine Similarity untuk TF-IDF dan Word2Vec:")
        if tfidf_source_matrix.shape[0] > 0 and tfidf_suspicious_matrix.shape[0] > 0:
            # Contoh Cosine Similarity dengan TF-IDF
            # Ambil satu dokumen mencurigakan dan bandingkan dengan semua sumber
            sample_suspicious_tfidf_vec = tfidf_suspicious_matrix[0:1] # Ambil baris pertama
            
            # Hitung cosine similarity dengan semua dokumen sumber
            tfidf_cosine_sims = cosine_similarity(sample_suspicious_tfidf_vec, tfidf_source_matrix)
            print(f"Cosine Similarity (TF-IDF) antara dokumen mencurigakan pertama dan sumber: {tfidf_cosine_sims[0][:5]} (hanya 5 pertama)")
        else:
            print("Tidak ada data TF-IDF untuk demonstrasi Cosine Similarity.")

        if word2vec_source_embeddings.shape[0] > 0 and word2vec_suspicious_embeddings.shape[0] > 0:
            # Contoh Cosine Similarity dengan Word2Vec embeddings
            sample_suspicious_w2v_vec = word2vec_suspicious_embeddings[0].reshape(1, -1) # Ambil baris pertama
            
            # Hitung cosine similarity dengan semua dokumen sumber
            w2v_cosine_sims = cosine_similarity(sample_suspicious_w2v_vec, word2vec_source_embeddings)
            print(f"Cosine Similarity (Word2Vec) antara dokumen mencurigakan pertama dan sumber: {w2v_cosine_sims[0][:5]} (hanya 5 pertama)")
        else:
            print("Tidak ada data Word2Vec untuk demonstrasi Cosine Similarity.")

    print("\nFeature extraction selesai!")