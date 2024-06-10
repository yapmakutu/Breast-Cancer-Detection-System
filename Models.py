import json
import gdown
import os

# Google Drive file IDs and local save names
models_info = {
    "unet_model": ("1ZekQ1o-WHF0pTeZOxZKOoEG8ULaFIbyh", "final_model.h5"),
    "trained_model": ("1JKGcZaMD8JUPlKmhtmA7mE-VeQMeirHg", "trained_model_size256.h5"),
    "knn_model": ("1shRne9zszfVKLR0p7RYAMcyhv5psDtiP", "knn_model.pkl"),
    "feature_len": ("1iOZINzghYES_-6QXm0Wl3hcoln9n0Gf-", "feature_len.pkl"),
    "scaler": ("1BztRc6AKCsS0lXIxOyQCCufV11YaIIVc", "scaler.pkl"),
    "pca": ("1_hPOKZdmWSAvvsfjpMmVyRZ9VBEibjIR", "pca.pkl")

}

# Directory to store downloaded models
download_path = "models"

# Create the directory if it does not exist
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Download models and save local paths
local_paths = {}
for model_name, (file_id, filename) in models_info.items():
    local_file_path = os.path.join(download_path, filename)
    gdown.download(f'https://drive.google.com/uc?id={file_id}', local_file_path, quiet=False)
    local_paths[f"{model_name}_path"] = local_file_path

# Update the config.json file with new paths
config_file_path = 'config.json'
if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)
else:
    config = {}

config.update(local_paths)

with open(config_file_path, 'w') as file:
    json.dump(config, file, indent=5)

print("Models have been downloaded and config.json updated.")
