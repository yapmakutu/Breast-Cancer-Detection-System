from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image


class ResultWindow(QDialog):
    def __init__(self, main_app, original_image_path=None, result_image_path=None, prediction=None):
        super().__init__()
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.button_exit = None
        self.button_upload_page = None
        self.label_diagnosis = None
        self.layout_results = None
        self.frame_results = None
        self.label_title = None
        self.main_app = main_app
        self.original_image_path = original_image_path
        self.result_image_path = result_image_path
        self.prediction = prediction
        self.initUI()
        self.update_diagnosis(prediction)
        self.showMaximized()

    def initUI(self):
        self.setWindowTitle("Result")
        self.setGeometry(100, 100, 1390, 768)  # Adjust these values as per your requirement
        self.setStyleSheet("background-color: #ADD8E6;")

        # Title Label
        self.label_title = QLabel("Result", self)
        self.label_title.setStyleSheet("color: black; font: 24pt 'Helvetica';")
        self.label_title.setAlignment(Qt.AlignCenter)

        # Results Frame
        self.frame_results = QFrame(self)
        self.frame_results.setStyleSheet("background-color: #323232;")
        self.layout_results = QHBoxLayout()

        # Load and display images
        self.load_and_display_image(self.original_image_path, Qt.AlignLeft)
        self.load_and_display_image(self.result_image_path, Qt.AlignRight)

        self.frame_results.setLayout(self.layout_results)

        # Diagnosis Label
        self.label_diagnosis = QLabel("...", self.frame_results)
        self.label_diagnosis.setStyleSheet("color: white; font: 16pt 'Helvetica';")
        self.label_diagnosis.setAlignment(Qt.AlignCenter)

        # Navigation Buttons
        self.button_upload_page = QPushButton("↩ Upload Page", self)
        self.button_upload_page.setStyleSheet("background-color: #FF69B4; color: black; font: 12pt 'Helvetica';")
        self.button_upload_page.clicked.connect(self.back_to_upload)

        self.button_exit = QPushButton("Exit ✖", self)
        self.button_exit.setStyleSheet("background-color: #FF69B4; color: black; font: 12pt 'Helvetica';")
        self.button_exit.clicked.connect(self.exit_app)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_title)
        main_layout.addWidget(self.frame_results)
        main_layout.addWidget(self.label_diagnosis)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_upload_page)
        button_layout.addWidget(self.button_exit)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def load_and_display_image(self, image_path, alignment):
        if image_path:
            img = Image.open(image_path)
            # Use Image.Resampling.LANCZOS instead of Image.ANTIALIAS
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            img.save(image_path)  # Optionally save the resized image back to disk

            pixmap = QPixmap(image_path)
            label_image = QLabel(self.frame_results)
            label_image.setPixmap(pixmap)
            label_image.setAlignment(alignment)
            self.layout_results.addWidget(label_image)

    def back_to_upload(self):
        self.close()
        self.main_app.open_image_loader()

    def exit_app(self):
        self.close()

    def update_diagnosis(self, prediction):
        diagnosis_text = "Not Cancer"  # Default text
        if isinstance(prediction, np.ndarray):
            # Process prediction from Deep Learning model
            diagnosis_text = str(prediction)  # Convert np.ndarray to string if necessary
        elif isinstance(prediction, str):
            # Direct label from Machine Learning model
            diagnosis_text = prediction
        self.label_diagnosis.setText(diagnosis_text)
