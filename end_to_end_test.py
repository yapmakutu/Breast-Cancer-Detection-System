import sys
import pytest
from PyQt5.QtWidgets import QApplication
from MainApplication import MainApplication, load_config

app = None  # QApplication için global değişken


@pytest.fixture(scope="session")
def app_fixture():
    global app
    if app is None:
        app = QApplication(sys.argv)  # QApplication sadece bir kere oluşturulur
    config_loader = load_config("config.json")
    main_app = MainApplication(config_loader)
    yield main_app
    main_app.close()


def test_deep_learning_workflow(app_fixture):
    test_image_path = r"C:\Users\AhmetSahinCAKIR\Desktop\Ahmet\Bitirme\Dataset_BUSI_with_GT\malignant\malignant (5).png"
    app_fixture.on_image_loaded(test_image_path, option=1)
    # Karşılaştırmayı küçük harfe çevirerek yap
    assert app_fixture.result_window.prediction.lower() in ['benign', 'malignant', 'Not Cancer']


def test_machine_learning_workflow(app_fixture):
    test_image_path = r"C:\Users\AhmetSahinCAKIR\Desktop\Ahmet\Bitirme\Dataset_BUSI_with_GT\malignant\malignant (5).png"
    app_fixture.on_image_loaded(test_image_path, option=2)
    # Karşılaştırmayı küçük harfe çevirerek yap
    assert app_fixture.result_window.prediction.lower() in ['benign', 'malignant', 'Not Cancer']


if __name__ == "__main__":
    # Pytest ile testleri çalıştır
    pytest.main(['-v', __file__])
