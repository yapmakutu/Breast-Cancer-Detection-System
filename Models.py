import json
import gdown
import os

# Google Drive file IDs and local save names
models_info = {
    "unet_model": ("1ZekQ1o-WHF0pTeZOxZKOoEG8ULaFIbyh", "final_model.h5"),
    "trained_model": ("1JKGcZaMD8JUPlKmhtmA7mE-VeQMeirHg", "trained_model_size256.h5"),
    "scaler": ("1R4o5yNKrs5jNwYUyTLzcZZhP03gXyUd4", "scaler.joblib"),
    "knn_model": ("1_x9apV_p8Vl-8oVIFSh1KIUbVwfX_b_5", "knn_model.joblib")
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
    json.dump(config, file, indent=4)

print("Models have been downloaded and config.json updated.")
