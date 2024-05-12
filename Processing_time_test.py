import unittest
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer, QTime

# Bu test doğru mu çözemedim. Yapmak istediğimi şey modellerden ne kadar sürede sonuç alma mı yoksa dümdüz butona
# tıkla ve ne kadar sürede atlıyor o adımı şeklinde mi?
class ProcessTest(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Deep Learning button
        self.button_dl = QPushButton('Deep Learning', self)
        self.button_dl.clicked.connect(lambda: self.process_image("dl"))
        layout.addWidget(self.button_dl)

        # Machine Learning button
        self.button_ml = QPushButton('Machine Learning', self)
        self.button_ml.clicked.connect(lambda: self.process_image("ml"))
        layout.addWidget(self.button_ml)

        self.label = QLabel('Processing time: None', self)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def process_image(self, model_type):
        self.timer = QTime()
        self.timer.start()
        # Simulate different processing delays for ML and DL
        if model_type == "ml":
            QTimer.singleShot(3000, lambda: self.image_processed(model_type))  # simulate 3 seconds for ML
        elif model_type == "dl":
            QTimer.singleShot(5000, lambda: self.image_processed(model_type))  # simulate 5 seconds for DL

    def image_processed(self, model_type):
        elapsed = self.timer.elapsed()
        self.label.setText(f"{model_type.upper()} Processing time: {elapsed} ms")
        self.processing_time = elapsed / 1000  # Convert ms to seconds for test evaluation
        self.model_type = model_type


class TestProcessTime(unittest.TestCase):
    def test_ml_processing_time(self):
        app = QApplication(sys.argv)
        process_test = ProcessTest()
        process_test.process_image("ml")

        # Wait for the QTimer to finish
        QTimer.singleShot(3100, app.quit)  # Wait a bit longer than the simulated processing time for ML
        app.exec_()

        # Check if the ML processing time is less than 40 seconds
        self.assertLess(process_test.processing_time, 40, "Machine Learning processing took too long.")
        self.assertEqual(process_test.model_type, "ml", "Test was not conducted on ML model")

    def test_dl_processing_time(self):
        app = QApplication(sys.argv)
        process_test = ProcessTest()
        process_test.process_image("dl")

        # Wait for the QTimer to finish
        QTimer.singleShot(5100, app.quit)  # Wait a bit longer than the simulated processing time for DL
        app.exec_()

        # Check if the DL processing time is less than 40 seconds
        self.assertLess(process_test.processing_time, 40, "Deep Learning processing took too long.")
        self.assertEqual(process_test.model_type, "dl", "Test was not conducted on DL model")


if __name__ == '__main__':
    unittest.main()
