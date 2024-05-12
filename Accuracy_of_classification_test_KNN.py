import numpy as np
import cv2
import os
import json
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


config = load_config("config.json")

knn_model = load(config["knn_model_path"])
scaler = load(config["scaler_path"])


def load_masked_images(folder):
    images = []
    labels = []
    label_mapping = {'benign': 0, 'malignant': 1, 'normal': 2}
    for label in label_mapping:
        dir_path = os.path.join(folder, label)
        for file in os.listdir(dir_path):
            if '_mask.png' in file:
                img_path = os.path.join(dir_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (256, 256))
                    images.append(img)
                    labels.append(label_mapping[label])
                else:
                    print(f"Failed to load image: {img_path}")
    print(f"Loaded {len(images)} images with labels.")
    return np.array(images), np.array(labels)


data_folder = r"C:\Users\AhmetSahinCAKIR\Desktop\Ahmet\Bitirme\Dataset_BUSI_with_GT"

images, labels = load_masked_images(data_folder)

# Görüntülerin piksel değerlerini 0-1 aralığına ölçeklendirme kısmı
images = images.astype('float32') / 255.0

# Görüntüleri düzleştirmem lazımmış modele göre
images_flat = images.reshape(images.shape[0], -1)

# %20 test
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)


scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn_model.fit(X_train_scaled, y_train)
y_pred = knn_model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
