from PyQt5.QtWidgets import (QDialog, QLabel, QPushButton, QFrame,
                             QVBoxLayout, QHBoxLayout, QFileDialog)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
from PIL import Image


def is_image_file(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    return os.path.splitext(file_path)[1].lower() in valid_extensions


class ImageLoaderApp(QDialog):
    def __init__(self, callback=None):
        super().__init__()
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.ok_button = None
        self.button_option_1 = None
        self.button_option_2 = None
        self.button_add = None
        self.drop_label = None
        self.drop_frame = None
        self.label_title = None
        self.setWindowTitle("Add Image")
        self.setGeometry(100, 100, 1390, 768)  # Adjust to center on the screen as needed
        self.callback = callback
        self.file_path = None
        self.option = None
        self.initUI()
        self.showMaximized()

    def initUI(self):
        self.setStyleSheet("background-color: #ADD8E6;")

        # Title Label
        self.label_title = QLabel("Add Image", self)
        self.label_title.setStyleSheet("color: black; font: 24pt 'Helvetica';")

        # Drop Frame
        self.drop_frame = QFrame(self)
        self.drop_frame.setStyleSheet("background-color: #323232;")
        self.drop_frame.setFixedSize(800, 550)

        # "Drop Image Here" Label
        self.drop_label = QLabel("Please select an Image", self.drop_frame)
        self.drop_label.setStyleSheet("color: black; font: 18pt 'Helvetica';")
        self.drop_label.setAlignment(Qt.AlignCenter)

        # "Add" Button
        self.button_add = QPushButton("Add", self.drop_frame)
        self.button_add.setStyleSheet("background-color: #FF69B4; color: black; font: 12pt 'Helvetica';")
        self.button_add.clicked.connect(self.open_file_dialog)

        # Option Buttons
        self.button_option_1 = QPushButton("Deep Learning", self)
        self.button_option_1.setStyleSheet("background-color: #FF69B4; color: black; font: 12pt 'Helvetica';")
        self.button_option_1.setFixedSize(200, 40)
        self.button_option_1.clicked.connect(lambda: self.set_option(1))

        self.button_option_2 = QPushButton("Machine Learning", self)
        self.button_option_2.setStyleSheet("background-color: #FF69B4; color: black; font: 12pt 'Helvetica';")
        self.button_option_2.setFixedSize(200, 40)
        self.button_option_2.clicked.connect(lambda: self.set_option(2))

        # "OK" Button
        self.ok_button = QPushButton("→", self)
        self.ok_button.setStyleSheet("background-color: #FF69B4; color: black; font: 12pt 'Helvetica';")
        self.ok_button.setFixedSize(50, 40)
        self.ok_button.clicked.connect(self.on_ok_pressed)

        self.layoutWidgets()

    def layoutWidgets(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_title, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.drop_frame, alignment=Qt.AlignCenter)

        # Drop frame layout
        drop_frame_layout = QVBoxLayout(self.drop_frame)
        drop_frame_layout.addWidget(self.drop_label, alignment=Qt.AlignCenter)
        drop_frame_layout.addWidget(self.button_add, alignment=Qt.AlignBottom | Qt.AlignLeft)

        # Option buttons layout
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.button_option_1)
        options_layout.addWidget(self.button_option_2)

        main_layout.addLayout(options_layout)
        main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.setLayout(main_layout)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image files (*.jpg *.jpeg *.png)")
        if file_path and is_image_file(file_path):
            # Önceki resmi kaldır
            self.remove_image()

            # Yeni dosyayı ekle
            self.file_path = file_path
            self.display_image(file_path)

    def remove_image(self):
        # Resmi kaldır ve dosya yolunu sıfırla
        self.drop_label.clear()
        self.file_path = None

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((512, 472), Image.LANCZOS)
        img.save(file_path)

        pixmap = QPixmap(file_path)
        self.drop_label.setPixmap(pixmap)
        self.drop_label.setScaledContents(True)

    def on_ok_pressed(self):
        if self.file_path and self.callback and self.option is not None:
            self.callback(self.file_path, self.option)
        self.close()

    def set_option(self, option):
        # Update the state of the selected button
        if option == 1:
            self.button_option_1.setStyleSheet("background-color: #4E342E; color: black; font: 12pt 'Helvetica';")
            self.button_option_2.setStyleSheet("background-color: #FF69B4; color: black; font: 12pt 'Helvetica';")
        elif option == 2:
            self.button_option_2.setStyleSheet("background-color: #4E342E; color: black; font: 12pt 'Helvetica';")
            self.button_option_1.setStyleSheet("background-color: #FF69B4; color: black; font: 12pt 'Helvetica';")

        self.option = option
