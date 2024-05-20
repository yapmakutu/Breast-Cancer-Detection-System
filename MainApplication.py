import json
import os
import tempfile
import tkinter as tk
import cv2
import numpy as np
from ImageLoaderApp import ImageLoaderApp
from DeepLearning import DeepLearning
from ResultWindow import ResultWindow
from MachineLearning import MachineLearning


def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


class MainApplication(tk.Tk):  # tk.Tk yerine tk.Toplevel kullanıldı
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Toplevel sınıfının __init__ metodunu çağırmak için super kullanıldı
        self.image_loader_app = None
        self.result_window = None
        # self.root = root
        # self.root.withdraw()
        unet_model_path = config["unet_model_path"]
        trained_model_path = config["trained_model_path"]
        scaler_path = config["scaler_path"]
        knn_model_path = config["knn_model_path"]
        self.deep_learning = DeepLearning(unet_model_path, trained_model_path)
        self.machine_learning = MachineLearning(scaler_path, knn_model_path)
        self.open_image_loader()

    def open_image_loader(self):
        if self.result_window is not None:
            self.result_window.destroy()
            self.result_window = None
        self.image_loader_app = ImageLoaderApp(master=self,
                                               callback=self.on_image_loaded)  # root yerine self kullanıldı
        self.image_loader_app.grab_set()

    def on_image_loaded(self, image_path, option):
        if option == 1:
            predicted_mask, prediction = self.deep_learning.segment_and_classify(image_path)
            self.image_loader_app.destroy()
            self.show_result_window(image_path, predicted_mask, prediction)
        elif option == 2:
            predicted_mask, _ = self.deep_learning.segment_and_classify(image_path)
            # Orijinal görüntüyü yükleyin
            original_image = cv2.imread(image_path)
            # Orijinal görüntü boyutlarını alın
            original_height, original_width = original_image.shape[:2]
            # Görüntüyü 256x256 piksel boyutlarına yeniden boyutlandırın
            resized_image = cv2.resize(original_image, (256, 256))
            # Yeniden boyutlandırılmış görüntüyü tekrar orijinal boyutlarına döndürün
            restored_image = cv2.resize(resized_image, (original_width, original_height))

            segmented_image, prediction = self.machine_learning.segment_and_classify(predicted_mask)
            self.image_loader_app.destroy()
            self.show_result_window(image_path, segmented_image, prediction)

    def show_result_window(self, original_image_path, result_image, prediction):
        if self.result_window is not None:
            self.result_window.destroy()
        if isinstance(result_image, np.ndarray):
            result_image = (result_image * 255).astype(np.uint8)
            result_image_path = os.path.join(tempfile.gettempdir(), "predicted_mask.png")
            cv2.imwrite(result_image_path, result_image)
        else:
            result_image_path = result_image
        self.result_window = ResultWindow(self, self, original_image_path=original_image_path,
                                          result_image_path=result_image_path,
                                          prediction=prediction)  # root yerine self kullanıldı
        self.result_window.grab_set()

    def run(self):
        self.mainloop()


config_loader = load_config("config.json")

if __name__ == "__main__":
    config_loader = load_config("config.json")
    app = MainApplication(config_loader)
    app.run()
