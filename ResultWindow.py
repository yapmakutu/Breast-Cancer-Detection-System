from PyQt5.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image
import os
import tempfile
from resultscreen import Ui_Dialog  # Import the generated UI class


class ResultWindow(QDialog, Ui_Dialog):
    def __init__(self, main_app, original_image_path=None, result_image_path=None, prediction=None):
        super().__init__()

        self.main_app = main_app
        self.original_image_path = original_image_path
        self.result_image_path = result_image_path
        self.prediction = prediction

        # Setup the UI
        self.setupUi(self)

        self.setFixedSize(1200, 800)
        self.initUI()
        self.update_diagnosis(prediction)

    def initUI(self):
        print("Step 12")
        self.layout_results = QHBoxLayout()  # Initialize the layout

        # Load and display images
        self.load_and_display_image(self.original_image_path, Qt.AlignLeft)
        self.load_and_display_image(self.result_image_path, Qt.AlignRight)

        self.frame_results.setLayout(self.layout_results)

        # Navigation Buttons
        self.button_upload_page.clicked.connect(self.back_to_upload)
        self.button_exit.clicked.connect(self.exit_app)

    def load_and_display_image(self, image_path, alignment):
        if image_path:
            try:
                img = Image.open(image_path)
                img = img.resize((511, 472), Image.Resampling.LANCZOS)  # Resize to fit the layout
                temp_image_path = os.path.join(tempfile.gettempdir(), os.path.basename(image_path))
                img.save(temp_image_path)  # Save the resized image

                pixmap = QPixmap(temp_image_path)
                label_image = QLabel(self.frame_results)
                label_image.setPixmap(pixmap)
                label_image.setAlignment(alignment)
                label_image.setFixedSize(511, 472)  # Set fixed size to avoid layout resizing
                self.layout_results.addWidget(label_image)
            except Exception as e:
                print(f"Error loading image: {e}")

    def back_to_upload(self):
        print("Back to upload")
        self.close()
        self.main_app.open_image_loader()

    def exit_app(self):
        print("Step 14")
        self.close()

    def update_diagnosis(self, prediction):
        diagnosis_text = "Not Cancer"  # Default text
        if isinstance(prediction, np.ndarray):
            diagnosis_text = str(prediction)  # Convert np.ndarray to string if necessary
        elif isinstance(prediction, str):
            diagnosis_text = prediction

        print("Step 13")
        self.label_diagnosis.setText(diagnosis_text)
        # Update diagnosis label style
        self.label_diagnosis.setStyleSheet(
            "background-color: white; color: black; font: 16pt 'Helvetica'; border: 2px solid black; padding: 5px;")
        self.label_diagnosis.setAlignment(Qt.AlignCenter)
