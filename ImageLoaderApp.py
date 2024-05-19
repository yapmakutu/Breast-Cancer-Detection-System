from PyQt5.QtWidgets import QDialog, QFileDialog, QLabel, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
import os
import test_rc
import sqlite3
from PyQt5.QtWidgets import QApplication

from PIL import Image


def is_image_file(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    return os.path.splitext(file_path)[1].lower() in valid_extensions


class ImageLoaderApp(QDialog):
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.file_path = None
        self.option = None

        # Load the UI file
        loadUi('firstscreen.ui', self)

        self.setFixedSize(1200, 800)

        # self.showMaximized()

        # initUI fonksiyonunu çağırın
        self.initUI()

    def initUI(self):
        # "Add" Button
        self.button_add.clicked.connect(self.open_file_dialog)
        self.button_option_1.clicked.connect(lambda: self.set_option(1))
        self.button_option_2.clicked.connect(lambda: self.set_option(2))
        self.ok_button.clicked.connect(self.on_ok_pressed)


    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image files (*.jpg *.jpeg *.png)")
        if file_path and is_image_file(file_path):
            # Önceki resmi kaldır
            self.remove_image()

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

