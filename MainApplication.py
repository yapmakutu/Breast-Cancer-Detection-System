import cProfile
import pstats
import sys
import json
import os
import tempfile
import threading  # Import threading module
from PyQt5.QtWidgets import QApplication, QMainWindow
import cv2
import numpy as np
import psutil  # Import psutil for resource monitoring
import time
from ImageLoaderApp import ImageLoaderApp
from ResultWindow import ResultWindow
from DeepLearning import DeepLearning
from MachineLearning import MachineLearning


def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


def display_usage(cpu_usage, mem_usage, bars):
    mem_percent = (mem_usage / 100)
    mem_bar = '█' * int(mem_percent * bars) + '-' * (bars - int(mem_percent * bars))

    cpu_percent = (cpu_usage / 100)
    cpu_bar = '█' * int(cpu_percent * bars) + '-' * (bars - int(cpu_percent * bars))


    print(f"\rCPU Usage: |{cpu_bar}| {cpu_usage:.2f}%   ", end="")
    print(f"MEM Usage: |{mem_bar}| {mem_usage:.2f}%   ", end="\r")
    sys.stdout.flush()


def resource_monitor():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        mem_usage = psutil.virtual_memory().percent
        display_usage(cpu_usage, mem_usage, 50)
        time.sleep(1)


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
        #print("Step 10")
        if option == 1:
            predicted_mask, prediction = self.deep_learning.segment_and_classify(image_path)
            self.show_result_window(image_path, predicted_mask, prediction)
        elif option == 2:
            segmented_image, prediction = self.machine_learning.segment_and_classify(image_path)
            self.show_result_window(image_path, segmented_image, prediction)

        if self.image_loader_app is not None:
            self.image_loader_app.close()

    def show_result_window(self, original_image_path, result_image, prediction):
        #print("Step 11")
        if isinstance(result_image, np.ndarray):
            result_image = (result_image * 255).astype(np.uint8)
            result_image_path = os.path.join(tempfile.gettempdir(), "predicted_mask.png")
            cv2.imwrite(result_image_path, result_image)
        else:
            result_image_path = result_image

        self.result_window = ResultWindow(self, original_image_path=original_image_path,
                                          result_image_path=result_image_path, prediction=prediction)
        self.result_window.show()


def main():
    # Start the resource monitor thread
    monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
    monitor_thread.start()

    app = QApplication(sys.argv)
    config_loader = load_config("config.json")
    main_app = MainApplication(config_loader)
    app.exec_()


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        main()
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
    results.dump_stats("result.prof")
