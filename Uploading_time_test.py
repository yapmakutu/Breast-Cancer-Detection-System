import unittest
import sys
import time
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import os


class ImageLoaderApp(QDialog):
    def __init__(self, callback=None, auto_image_path=None):
        super().__init__()
        self.setWindowTitle("Add Image")
        self.setGeometry(100, 100, 1390, 768)
        self.callback = callback
        self.file_path = None
        self.option = None
        self.initUI(auto_image_path)

    def initUI(self, auto_image_path):
        self.setStyleSheet("background-color: #ADD8E6;")
        self.label_title = QLabel("Add Image", self)
        self.label_title.setStyleSheet("color: black; font: 24pt 'Helvetica';")
        self.drop_label = QLabel("Please select an Image", self)
        self.drop_label.setStyleSheet("color: black; font: 18pt 'Helvetica';")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFixedSize(800, 550)
        self.layoutWidgets()
        if auto_image_path:
            self.display_image(auto_image_path)

    def layoutWidgets(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_title, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.drop_label, alignment=Qt.AlignCenter)
        self.setLayout(main_layout)

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((self.drop_label.width(), self.drop_label.height()), Image.LANCZOS)
        img_path_temp = os.path.join(os.path.dirname(file_path), "temp_image_display.jpg")
        img.save(img_path_temp)
        pixmap = QPixmap(img_path_temp)
        self.drop_label.setPixmap(pixmap)
        self.drop_label.setScaledContents(True)
        self.file_path = file_path


class TestImageLoader(unittest.TestCase):
    def test_image_loading(self):
        app = QApplication(sys.argv)
        image_path = r"C:\Users\AhmetSahinCAKIR\Desktop\Test_Images\benign (3).png"
        start_time = time.time()
        image_loader = ImageLoaderApp(auto_image_path=image_path)
        image_loader.display_image(image_path)
        load_time = time.time() - start_time

        print(f"Loading time: {load_time:.2f} seconds")
        self.assertTrue(load_time < 40, "Image loading took too long, exceeding 40 seconds.")


if __name__ == '__main__':
    unittest.main()
