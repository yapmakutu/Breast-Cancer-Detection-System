from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image
import test_rc
import sqlite3
from PyQt5.uic import loadUi


class ResultWindow(QDialog):
    def __init__(self, main_app, original_image_path=None, result_image_path=None, prediction=None):
        super().__init__()

        self.layout_results = None
        self.main_app = main_app
        self.original_image_path = original_image_path
        self.result_image_path = result_image_path
        self.prediction = prediction

        loadUi('resultscreen.ui', self)

        self.setFixedSize(1200, 800)
        self.initUI()
        self.update_diagnosis(prediction)

    def initUI(self):

        # Load and display images
        self.load_and_display_image(self.original_image_path, Qt.AlignLeft)
        self.load_and_display_image(self.result_image_path, Qt.AlignRight)

        self.frame_results.setLayout(self.layout_results)

        # Navigation Buttons
        self.button_upload_page.clicked.connect(self.back_to_upload)
        self.button_exit.clicked.connect(self.exit_app)

    def load_and_display_image(self, image_path, alignment):
        if image_path:
            img = Image.open(image_path)
            img = img.resize((511, 472), Image.Resampling.LANCZOS)  # Resize to 511x472
            img.save(image_path)  # Optionally save the resized image back to disk

            pixmap = QPixmap(image_path)
            label_image = QLabel(self.frame_results)
            label_image.setPixmap(pixmap)
            label_image.setAlignment(alignment)
            label_image.setFixedSize(511, 472)  # Set fixed size to avoid layout resizing
            self.layout_results.addWidget(label_image)

    def back_to_upload(self):
        self.close()
        self.main_app.open_image_loader()

    def exit_app(self):
        self.close()

    def update_diagnosis(self, prediction):
        diagnosis_text = "Not Cancer"  # Default text
        if isinstance(prediction, np.ndarray):
            diagnosis_text = str(prediction)  # Convert np.ndarray to string if necessary
        elif isinstance(prediction, str):
            diagnosis_text = prediction

        self.label_diagnosis.setText(diagnosis_text)
        # Update diagnosis label style
        self.label_diagnosis.setStyleSheet("background-color: #323232; color: white; font: 16pt 'Helvetica';")
        self.label_diagnosis.setText(diagnosis_text)
