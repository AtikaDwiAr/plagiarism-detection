import os
import gdown
import zipfile

# Dataset: https://drive.google.com/file/d/1Rlg4L4ieMkHihTXNuABoFyh6lD5wLCl3/view?usp=drive_link

# Unduh ZIP dari Google Drive
file_id = '1Rlg4L4ieMkHihTXNuABoFyh6lD5wLCl3'  # Ganti dengan ID ZIP dataset
zip_name = 'pan_dataset.zip'
zip_url = f'https://drive.google.com/uc?id={file_id}'

print("Mengunduh dataset dari Google Drive...")
gdown.download(zip_url, zip_name, quiet=False)

# Ekstrak ZIP
extract_path = 'pan_dataset'

print("Mengekstrak ZIP...")
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Path Folder
source_path = os.path.join(extract_path, 'external-detection-corpus', 'source-document', 'part1')
suspicious_path = os.path.join(extract_path, 'external-detection-corpus', 'suspicious-document', 'part1')
