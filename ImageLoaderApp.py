import sys
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import os
from PIL import Image
from firstscreen import Ui_Dialog  # Ensure this import matches the converted .py file


def is_image_file(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    return os.path.splitext(file_path)[1].lower() in valid_extensions


class ImageLoaderApp(QDialog, Ui_Dialog):
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.file_path = None
        self.option = None

        # Setup the UI
        self.setupUi(self)

        self.initUI()

    def initUI(self):
        self.button_add.clicked.connect(self.handle_add_button_click)
        self.button_option_1.clicked.connect(lambda: self.set_option(1))
        self.button_option_2.clicked.connect(lambda: self.set_option(2))
        self.ok_button.clicked.connect(self.handle_ok_button_click)

    def handle_add_button_click(self):
        self.button_add.setStyleSheet(
            "background-color: #FFD700; color: black; font: 12pt 'Helvetica';")  # Change to a highlight color
        QTimer.singleShot(200, self.reset_add_button_color)  # Change color back after 200 milliseconds
        self.open_file_dialog()

    def reset_add_button_color(self):
        self.button_add.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
                                      "color: white;\n"
                                      "text-align: center;\n"
                                      "border-radius: 19px;\n"
                                      "width: 200px;\n"
                                      "height: 100px;\n")

    def handle_ok_button_click(self):
        self.ok_button.setStyleSheet(
            "background-color: #FFD700; color: black; font: 12pt 'Helvetica';")  # Change to a highlight color
        QTimer.singleShot(200, self.reset_ok_button_color)  # Change color back after 200 milliseconds
        self.on_ok_pressed()

    def reset_ok_button_color(self):
        self.ok_button.setStyleSheet("font: 75 14pt 'Goudy Old Style'; color: white;")

    def open_file_dialog(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image files (*.jpg *.jpeg *.png)")
            if file_path and is_image_file(file_path):
                self.remove_image()
                self.file_path = file_path
                self.display_image(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error5", str(e))

    def remove_image(self):
        self.drop_label.clear()
        self.file_path = None

    def display_image(self, file_path):
        try:
            img = Image.open(file_path)
            img = img.resize((581, 541), Image.LANCZOS)
            img.save(file_path)
            pixmap = QPixmap(file_path)
            self.drop_label.setPixmap(pixmap)
            self.drop_label.setScaledContents(True)
        except Exception as e:
            QMessageBox.critical(self, "Error6", str(e))

    def on_ok_pressed(self):
        try:
            if self.file_path and self.callback and self.option is not None:
                self.callback(self.file_path, self.option)
            else:
                QMessageBox.warning(self, "Warning", "Missing file path or option")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error7", str(e))

    def set_option(self, option):
        if option == 1:
            self.button_option_1.setStyleSheet("background-color: #333333; color: white; font: 14pt 'Goudy Old Style';")
            self.button_option_2.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
                                               "color: rgb(255, 255, 255);\n"
                                               "border: 5px solid white; border-radius: 10px;")
        elif option == 2:
            self.button_option_2.setStyleSheet("background-color: #333333; color: white; font: 14pt 'Goudy Old Style';")
            self.button_option_1.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
                                               "color: rgb(255, 255, 255);\n"
                                               "border: 5px solid white; border-radius: 10px;")

        self.option = option


if __name__ == "__main__":
    def test_callback(file_path, option):
        print(f"Callback invoked with file_path: {file_path} and option: {option}")


    app = QApplication(sys.argv)
    mainWindow = ImageLoaderApp(callback=test_callback)
    mainWindow.show()
    sys.exit(app.exec_())
