import sys
import os
import cv2
import joblib
import numpy as np
from skimage import feature, measure
from skimage.measure import regionprops


class MachineLearning:
    def __init__(self, scaler_path, knn_model_path, pca_path, feature_len_path):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        scaler_path = os.path.join(base_path, scaler_path)
        pca_path = os.path.join(base_path, pca_path)
        knn_model_path = os.path.join(base_path, knn_model_path)
        feature_len_path = os.path.join(base_path, feature_len_path)

        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.knn_classifier = joblib.load(knn_model_path)
        self.max_len = joblib.load(feature_len_path)

    @staticmethod
    def extract_lbp_features(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.uint8)
        lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        return hist

    def segment_and_classify(self, image_path):
        ultrasound_image = cv2.imread(image_path)
        if ultrasound_image is None:
            return None, "Image could not be uploaded correctly."

        if np.max(ultrasound_image) == 0:
            return None, "No Cancer"

        # K-means segmentasyonu
        if len(ultrasound_image.shape) == 3:
            k_means_segmented = self.k_means_segmentation(ultrasound_image, K=2)
            gray_image = cv2.cvtColor(k_means_segmented, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = ultrasound_image

        # Görüntü işleme adımları
        salt_img_filtered = cv2.medianBlur(gray_image, ksize=3)
        gaussian_img_filtered = cv2.GaussianBlur(salt_img_filtered, (3, 3), 0)
        img_filtered = cv2.bilateralFilter(gaussian_img_filtered, d=9, sigmaColor=75, sigmaSpace=75)
        _, thresholded_img = cv2.threshold(img_filtered, 127, 255, cv2.THRESH_BINARY)

        img = np.array(thresholded_img, dtype=np.float64)
        IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
        IniLSF[30:80, 30:80] = -1
        IniLSF = -IniLSF

        # Chan-Vese segmentasyonu
        LSF = self.chan_vese_segmentation(IniLSF, img)

        contour_mask = np.zeros_like(LSF, dtype=np.uint8)
        contour_mask[LSF > 0] = 255

        if len(ultrasound_image.shape) == 3:
            result_contour = ultrasound_image.copy()
        else:
            result_contour = cv2.cvtColor(ultrasound_image, cv2.COLOR_GRAY2BGR)

        result_contour[contour_mask > 0] = [255, 0, 0]
        edge_image = MachineLearning.detect_and_draw_circle(result_contour)

        mask = LSF > 0
        segmented_image = np.zeros_like(ultrasound_image)
        segmented_image[mask] = ultrasound_image[mask]

        segmented_lbp_features = self.extract_lbp_features(ultrasound_image)

        if np.all(segmented_lbp_features == 0):
            return None, "No Cancer"

        if len(segmented_lbp_features) < self.max_len:
            segmented_features_padded = np.pad(segmented_lbp_features, (0, self.max_len - len(segmented_lbp_features)),
                                               'constant')
        else:
            segmented_features_padded = segmented_lbp_features[:self.max_len]
        segmented_features_scaled = self.scaler.transform([segmented_features_padded])
        segmented_features_pca = self.pca.transform(segmented_features_scaled)
        predicted_label = self.knn_classifier.predict(segmented_features_pca)[0]

        if predicted_label == 2 and not self.is_normal_mask(ultrasound_image):
            predicted_label = 1  # Adjust to benign if mask is not normal
        if predicted_label != 0 and self.is_malignant_mask(ultrasound_image):
            predicted_label = 0  # Adjust to malignant if mask is malignant

        label_mapping = {0: 'Malignant', 1: 'Benign', 2: 'No Cancer'}
        return edge_image, label_mapping[predicted_label]

    @staticmethod
    def is_normal_mask(image, low_threshold=0.005, high_threshold=0.02, max_region_area=100, max_total_regions=10):
        white_pixel_ratio = np.sum(image > 0) / (image.shape[0] * image.shape[1])
        if white_pixel_ratio < low_threshold:
            return True
        labeled_mask = measure.label(image > 0)
        regions = regionprops(labeled_mask)
        small_region_count = 0
        for region in regions:
            if region.area < max_region_area:
                small_region_count += 1
            if small_region_count > max_total_regions:
                return False
        return white_pixel_ratio < high_threshold and small_region_count > 0

    @staticmethod
    def is_malignant_mask(image, min_white_threshold=0.05, min_region_area=500):
        white_pixel_ratio = np.sum(image > 0) / (image.shape[0] * image.shape[1])
        if white_pixel_ratio < min_white_threshold:
            return False
        labeled_mask = measure.label(image > 0)
        regions = regionprops(labeled_mask)
        for region in regions:
            if region.area >= min_region_area:
                return True
        return False

    def chan_vese_segmentation(self, LSF, img, mu=1, nu=0.003 * 255 * 255, num_iter=20, epsilon=1, step=0.1):
        for i in range(1, num_iter):
            LSF = self.CV(LSF, img, mu, nu, epsilon, step)
        return LSF

    @staticmethod
    def CV(LSF, img, mu, nu, epsilon, step):
        Drc = (epsilon / np.pi) / (epsilon * 2 + LSF * 2 + 1e-10)
        Hea = 0.5 * (1 + (2 / np.pi) * MachineLearning.mat_math(LSF / epsilon, "atan"))

        Iy, Ix = np.gradient(LSF)
        s = MachineLearning.mat_math(Ix * 2 + Iy * 2, "sqrt")
        Nx = Ix / (s + 1e-6)
        Ny = Iy / (s + 1e-6)
        Mxx, Nxx = np.gradient(Nx)
        Nyy, Myy = np.gradient(Ny)
        curvature = Nxx + Nyy
        curvature = np.clip(curvature, -1e10, 1e10)

        Length = nu * Drc * curvature

        Lap = cv2.Laplacian(LSF, cv2.CV_64F)
        Penalty = mu * (Lap - curvature)

        s1 = Hea * img
        s2 = (1 - Hea) * img
        s3 = 1 - Hea
        C1 = s1.sum() / (Hea.sum() + 1e-10)
        C2 = s2.sum() / (s3.sum() + 1e-10)
        CVterm = Drc * (-1 * (img - C1) * 2 + (img - C2) * 2)

        LSF = LSF + step * (Length + Penalty + CVterm)
        return LSF

    @staticmethod
    def k_means_segmentation(image, K=2):
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_image = res.reshape(image.shape)

        return segmented_image

    @staticmethod
    def mat_math(input_array, operation):
        input_array = np.clip(input_array, -1e10, 1e10)
        if operation == "atan":
            return np.arctan(input_array)
        elif operation == "sqrt":
            input_array[input_array < 0] = 0
            return np.sqrt(input_array)
        else:
            raise ValueError("Unsupported operation")

    @staticmethod
    def detect_and_draw_circle(image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        if gray_image.dtype != np.uint8:
            gray_image = gray_image.astype(np.uint8)

        edges = cv2.Canny(gray_image, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)

            result_image = image.copy()
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)

            return result_image
        else:
            return image
