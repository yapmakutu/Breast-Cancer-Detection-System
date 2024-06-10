import ctypes
import logging
import sys
import json
import os
import tempfile
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import cv2
import numpy as np
from ImageLoaderApp import ImageLoaderApp
from ResultWindow import ResultWindow
from DeepLearning import DeepLearning
from MachineLearning import MachineLearning

# Hide the console window
if os.name == 'nt':
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# Configure logging
log_file_path = os.path.join(tempfile.gettempdir(), 'app_debug.log')
logging.basicConfig(filename=log_file_path, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def except_hook(cls, exception, traceback):
    logging.error("Unhandled exception", exc_info=(cls, exception, traceback))
    QMessageBox.critical(None, "Error", str(exception))


sys.excepthook = except_hook

logging.info("Application started")


def load_config(config_file):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, config_file)
    logging.debug(f"Loading config from: {config_path}")
    with open(config_path, "r") as file:
        return json.load(file)


class MainApplication(QMainWindow):
    def __init__(self, config):
        super(MainApplication, self).__init__()
        self.setWindowTitle("Breast Cancer Detection System")

        # Initialize models
        try:
            self.deep_learning = DeepLearning(config["unet_model_path"], config["trained_model_path"])
            logging.info("Deep learning models initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing deep learning models: {e}")
            QMessageBox.critical(self, "Error", f"Error initializing deep learning models: {e}")
            raise

        try:
            self.machine_learning = MachineLearning(config["scaler_path"], config["knn_model_path"], config["pca_path"],
                                                    config["feature_len"])
            logging.info("Machine learning models initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing machine learning models: {e}")
            QMessageBox.critical(self, "Error", f"Error initializing machine learning models: {e}")
            raise

        self.image_loader_app = None
        self.result_window = None

        self.open_image_loader()

    def open_image_loader(self):
        self.image_loader_app = ImageLoaderApp(callback=self.on_image_loaded)
        self.image_loader_app.show()

    def on_image_loaded(self, image_path, option):
        try:
            logging.debug(f"Image loaded: {image_path}, Option: {option}")
            if option == 1:
                try:
                    predicted_mask, prediction = self.deep_learning.segment_and_classify(image_path)
                    self.show_result_window(image_path, predicted_mask, prediction)
                except Exception as e:
                    logging.error(f"Option 1 Deep Learning Error: {e}")
                    raise ValueError(f"Option 1 Deep Learning Error: {e}")
            elif option == 2:
                try:
                    predicted_mask, _ = self.deep_learning.segment_and_classify(image_path)
                except Exception as e:
                    logging.error(f"Option 2 Deep Learning Error: {e}")
                    raise ValueError(f"Option 2 Deep Learning Error: {e}")
                try:
                    _, prediction = self.machine_learning.segment_and_classify(predicted_mask)
                    self.show_result_window(image_path, predicted_mask, prediction)
                except Exception as e:
                    logging.error(f"Machine Learning Error: {e}")
                    raise ValueError(f"Machine Learning Error: {e}")
            if self.image_loader_app is not None:
                self.image_loader_app.close()
        except Exception as e:
            logging.error(f"Error in on_image_loaded: {e}")
            QMessageBox.critical(self, "Error in on_image_loaded", str(e))

    def show_result_window(self, original_image_path, result_image, prediction):
        try:
            if isinstance(result_image, np.ndarray):
                result_image = (result_image * 255).astype(np.uint8)
                result_image_path = os.path.join(tempfile.gettempdir(), "predicted_mask.png")
                if not cv2.imwrite(result_image_path, result_image):
                    raise ValueError(f"Result page Could not write image to path: {result_image_path}")
            else:
                result_image_path = result_image

            self.result_window = ResultWindow(self, original_image_path=original_image_path,
                                              result_image_path=result_image_path, prediction=prediction)
            self.result_window.show()
        except Exception as e:
            logging.error(f"Error in show_result_window: {e}")
            QMessageBox.critical(self, "Error2", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        config_loader = load_config("config.json")
        main_app = MainApplication(config_loader)
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Error in main: {e}")
        QMessageBox.critical(None, "Error4", str(e))
