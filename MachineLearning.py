import math
import cv2
import numpy as np
from skimage import feature
import joblib
import matplotlib.pyplot as plt
import os


class MachineLearning:
    def __init__(self, scaler_path, knn_model_path):
        self.scaler = joblib.load(scaler_path)
        self.knn_classifier = joblib.load(knn_model_path)

    @staticmethod
    def extract_lbp_features(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Görüntüyü tam sayı türüne çevirin
        image = image.astype(np.uint8)

        lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        return hist

    def segment_and_classify(self, ultrasound_image):
        if ultrasound_image is None:
            return None, "Görüntü yüklenemedi"
        if len(ultrasound_image.shape) == 3:
            k_means_segmented = self.k_means_segmentation(ultrasound_image, K=2)
            gray_image = cv2.cvtColor(k_means_segmented, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = ultrasound_image

        salt_img_filtered = cv2.medianBlur(gray_image, ksize=3)
        gaussian_img_filtered = cv2.GaussianBlur(salt_img_filtered, (3, 3), 0)
        img_filtered = cv2.bilateralFilter(gaussian_img_filtered, d=9, sigmaColor=75, sigmaSpace=75)
        _, thresholded_img = cv2.threshold(img_filtered, 127, 255, cv2.THRESH_BINARY)

        img = np.array(thresholded_img, dtype=np.float64)
        IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
        IniLSF[30:80, 30:80] = -1
        IniLSF = -IniLSF

        LSF = self.chan_vese_segmentation(IniLSF, img)

        # Zero array icine LSF value olan yerleri doldurma
        contour_mask = np.zeros_like(LSF, dtype=np.uint8)
        contour_mask[LSF > 0] = 255

        # Orijinal görüntü üzerine active contour sınırlarını ekle
        if len(ultrasound_image.shape) == 3:
            result_contour = ultrasound_image.copy()
        else:
            result_contour = cv2.cvtColor(ultrasound_image, cv2.COLOR_GRAY2BGR)

        # Active contour sınırlarını kırmızı renkte ekle
        result_contour[contour_mask > 0] = [255, 0, 0]
        edge_image = MachineLearning.detect_and_draw_circle(result_contour)

        mask = LSF > 0
        segmented_image = np.zeros_like(ultrasound_image)
        segmented_image[mask] = ultrasound_image[mask]
        segmented_lbp_features = self.extract_lbp_features(ultrasound_image)

        segmented_features_scaled = self.scaler.transform([segmented_lbp_features])
        predicted_label = self.knn_classifier.predict(segmented_features_scaled)[0]
        label_mapping = {0: 'Malignant', 1: 'Benign', 2: 'Normal'}
        return edge_image, label_mapping[predicted_label]

    def chan_vese_segmentation(self, LSF, img, mu=1, nu=0.003 * 255 * 255, num_iter=20, epsilon=1, step=0.1):
        for i in range(1, num_iter):
            LSF = self.CV(LSF, img, mu, nu, epsilon, step)
        return LSF

    def k_means_segmentation(self, image, K=2):
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))

        return segmented_image

    @staticmethod
    def mat_math(input_array, operation):

        if operation == "atan":
            return np.arctan(input_array)
        elif operation == "sqrt":
            return np.sqrt(input_array)
        else:
            raise ValueError("Unsupported operation")

    @staticmethod
    def CV(LSF, img, mu, nu, epsilon, step):
        # Aktivasyon fonksiyonu (Heaviside fonksiyonu)
        Drc = (epsilon / math.pi) / (epsilon ** 2 + LSF ** 2)
        Hea = 0.5 * (1 + (2 / math.pi) * MachineLearning.mat_math(LSF / epsilon, "atan"))

        # Gradient hesaplamaları
        Iy, Ix = np.gradient(LSF)
        s = MachineLearning.mat_math(Ix ** 2 + Iy ** 2, "sqrt")
        Nx = Ix / (s + 1e-6)
        Ny = Iy / (s + 1e-6)
        Mxx, Nxx = np.gradient(Nx)
        Nyy, Myy = np.gradient(Ny)
        curvature = Nxx + Nyy
        Length = nu * Drc * curvature

        # Laplacian ve Penalty hesaplamaları
        Lap = cv2.Laplacian(LSF, cv2.CV_64F)
        Penalty = mu * (Lap - curvature)

        # Sınıf merkezi (C1 ve C2) ve CV terimi hesaplamaları
        s1 = Hea * img
        s2 = (1 - Hea) * img
        s3 = 1 - Hea
        C1 = s1.sum() / Hea.sum()
        C2 = s2.sum() / s3.sum()
        CVterm = Drc * (-1 * (img - C1) ** 2 + (img - C2) ** 2)

        # LSF'nin güncellenmesi
        LSF = LSF + step * (Length + Penalty + CVterm)
        return LSF

    @staticmethod
    def detect_and_draw_circle(image):
        # giriş görüntüsünü 8-bit gri tonlamalı hale getirin
        if len(image.shape) == 3:
            #print("Hello0")
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            #print("Hello1")
            gray_image = image

        if gray_image.dtype != np.uint8:
            #print("Hello2")
            gray_image = gray_image.astype(np.uint8)

        # kenarları bulma
        edges = cv2.Canny(gray_image, 30, 100)

        # contour'ları bulma
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            #print("Hello3")
            # en büyük konturun bulunması
            largest_contour = max(contours, key=cv2.contourArea)

            # contour'un çevresine bir çember çizme
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)

            # görüntü üzerine çemberi çizme
            result_image = image.copy()
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)

        return result_image
        #else:
        #    return image  # contour bulunamazsa orijinal görüntüyü döndür

