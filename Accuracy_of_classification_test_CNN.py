import numpy as np
import cv2
import os
import json
from keras.models import load_model
from sklearn.model_selection import train_test_split


def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


config = load_config("config.json")
cnn_model = load_model(config["trained_model_path"])


def load_masked_images(folder):
    images = []
    labels = []
    label_mapping = {'benign': 0, 'malignant': 1}
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


def predict_with_cnn(model, images, threshold=0.6, batch_size=32):
    images = images.reshape((-1, 256, 256, 1))
    images = images / 255.0  # Normalize the images
    predictions = model.predict(images, batch_size=batch_size)
    diagnosis_labels = [1 if pred > threshold else 0 for pred in predictions]
    # 1 if the probability of 'malignant' is greater than 0.6
    return np.array(diagnosis_labels)


folder = r"C:\Users\AhmetSahinCAKIR\Desktop\Ahmet\Bitirme\Dataset_BUSI_with_GT"
images, labels = load_masked_images(folder)
_, test_images, _, test_labels = train_test_split(images, labels, test_size=0.2)
deep_predictions = predict_with_cnn(cnn_model, test_images)


def calculate_performance(true_labels, predictions):
    # Convert true_labels to integers if necessary
    true_labels = np.array(true_labels, dtype=int)  # Ensure labels are integers
    predictions = np.array(predictions, dtype=int)  # Ensure predictions are integers

    # Calculate accuracy
    accuracy = np.mean(true_labels == predictions)
    return accuracy


predicted_labels = predict_with_cnn(cnn_model, test_images)

dl_accuracy = calculate_performance(test_labels, predicted_labels)
dl_accuracy_percent = dl_accuracy * 100
print(f'Deep Learning Model (CNN) Accuracy: {dl_accuracy_percent:.2f}%')
