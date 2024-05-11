import numpy as np
import cv2
import json
from keras.models import load_model
from sklearn.metrics import accuracy_score
import joblib
from skimage.feature import local_binary_pattern


# Konfigürasyon dosyasını yükleme
def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


config = load_config("config.json")

# Deep Learning Modelleri
unet_model = load_model(config["unet_model_path"])
cnn_model = load_model(config["trained_model_path"])

# Machine Learning Modeli
ml_model = joblib.load(config["knn_model_path"])
scaler = joblib.load(config["scaler_path"])


# Test verilerini yükleme
def load_test_data(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))  # Eğer gerekirse boyutlandırma
        images.append(img)
    return np.array(images)


test_image_paths = ['test1.jpg', 'test2.jpg']  # Örnek yollar
test_images = load_test_data(test_image_paths)
true_labels = np.array([1, 0])  # Gerçek etiketler


# CNN tahmin fonksiyonu
def predict_with_cnn(model, images):
    # Boyut ekleme gerekiyorsa
    images = images / 255.0  # Normalize etme (gerekiyorsa)
    return model.predict(images).argmax(axis=1)


deep_predictions1 = predict_with_cnn(deep_model1, test_images)
deep_predictions2 = predict_with_cnn(deep_model2, test_images)


# LBP özellikleri çıkarma
def extract_lbp_features(image, P=8, R=1):
    lbp_image = local_binary_pattern(image, P, R, method='uniform')
    n_bins = int(lbp_image.max() + 1)
    hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_features_and_predict(images, model, scaler):
    features = [extract_lbp_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images]
    scaled_features = scaler.transform(features)
    return model.predict(scaled_features)


ml_predictions = extract_features_and_predict(test_images, ml_model, scaler)


# Performans değerlendirme
def calculate_performance(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


accuracy1 = calculate_performance(true_labels, deep_predictions1)
accuracy2 = calculate_performance(true_labels, deep_predictions2)
ml_accuracy = calculate_performance(true_labels, ml_predictions)

# Performansları kaydetme
with open('model_performances.txt', 'w') as file:
    file.write(f'Deep Learning Model 1 Accuracy: {accuracy1}\n')
    file.write(f'Deep Learning Model 2 Accuracy: {accuracy2}\n')
    file.write(f'Machine Learning Model Accuracy: {ml_accuracy}\n')
