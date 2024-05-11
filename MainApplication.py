import sys
import json
import os
import tempfile
from PyQt5.QtWidgets import QApplication, QMainWindow
import cv2
import numpy as np
from ImageLoaderApp import ImageLoaderApp
from ResultWindow import ResultWindow
from DeepLearning import DeepLearning
from MachineLearning import MachineLearning

#projeyi bozmayalım lütfennnn

def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


class MainApplication(QMainWindow):
    def __init__(self, config):
        super(MainApplication, self).__init__()
        self.setWindowTitle("Breast Cancer Detection System")

        # Initialize models
        self.deep_learning = DeepLearning(config["unet_model_path"], config["trained_model_path"])
        self.machine_learning = MachineLearning(config["scaler_path"], config["knn_model_path"])

        self.image_loader_app = None
        self.result_window = None

        self.open_image_loader()

    def open_image_loader(self):
        self.image_loader_app = ImageLoaderApp(callback=self.on_image_loaded)
        self.image_loader_app.show()

    def on_image_loaded(self, image_path, option):
        if option == 1:
            predicted_mask, prediction = self.deep_learning.segment_and_classify(image_path)
            self.show_result_window(image_path, predicted_mask, prediction)
        elif option == 2:
            segmented_image, prediction = self.machine_learning.segment_and_classify(image_path)
            self.show_result_window(image_path, segmented_image, prediction)

        if self.image_loader_app is not None:
            self.image_loader_app.close()

    def show_result_window(self, original_image_path, result_image, prediction):
        if isinstance(result_image, np.ndarray):
            result_image = (result_image * 255).astype(np.uint8)
            result_image_path = os.path.join(tempfile.gettempdir(), "predicted_mask.png")
            cv2.imwrite(result_image_path, result_image)
        else:
            result_image_path = result_image

        self.result_window = ResultWindow(self, original_image_path=original_image_path,
                                          result_image_path=result_image_path, prediction=prediction)
        self.result_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    config_loader = load_config("config.json")
    main_app = MainApplication(config_loader)
    sys.exit(app.exec_())

