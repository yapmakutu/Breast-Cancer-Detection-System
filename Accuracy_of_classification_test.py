import numpy as np
import cv2
import os
import json
from skimage import feature
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


config = load_config("config.json")
cnn_model = load_model(config["trained_model_path"])
ml_model = joblib.load(config["knn_model_path"])
scaler = joblib.load(config["scaler_path"])


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

    print(f"Loaded {len(images)} images with labels.")
    return np.array(images), np.array(labels)


folder = r'C:\Users\AhmetSahinCAKIR\Desktop\Ahmet\Bitirme\Dataset_BUSI_with_GT'
images, labels = load_masked_images(folder)
test_size_ratio = 0.2  # 20% test
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size_ratio,
                                                                        random_state=42)


def predict_with_cnn(model, images):
    images = images.reshape(images.shape[0], 256, 256, 1)
    images = images / 255.0
    return model.predict(images).argmax(axis=1)


deep_predictions = predict_with_cnn(cnn_model, test_images)


def extract_lbp_features(image):
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    return hist


def extract_features_and_predict(images, model, scaler):
    features = np.array([extract_lbp_features(img) for img in images])
    scaled_features = scaler.transform(features)
    return model.predict(scaled_features)


ml_predictions = extract_features_and_predict(test_images, ml_model, scaler)


def calculate_performance(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


dl_accuracy = calculate_performance(test_labels, deep_predictions)
ml_accuracy = calculate_performance(test_labels, ml_predictions)

dl_accuracy_percent = dl_accuracy * 100
ml_accuracy_percent = ml_accuracy * 100

print(f'Deep Learning Model (CNN) Accuracy: {dl_accuracy_percent:.2f}%')
print(f'Machine Learning Model (KNN) Accuracy: {ml_accuracy_percent:.2f}%')
