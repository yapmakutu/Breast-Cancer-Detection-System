from PyQt5 import QtCore, QtGui, QtWidgets
import test_rc


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(0, 0, 1200, 800))
        self.widget.setAcceptDrops(True)
        self.widget.setStyleSheet("QWidget#widget{\n"
                                  "\n"
                                  "border-image: url(:/clippart/78505c683664dc60f3b10524f15de2eb.jpg);\n"
                                  "}\n"
                                  "")
        self.widget.setObjectName("widget")
        self.button_option_2 = QtWidgets.QPushButton(self.widget)
        self.button_option_2.setGeometry(QtCore.QRect(510, 570, 271, 71))
        self.button_option_2.setAcceptDrops(False)
        self.button_option_2.setStatusTip("")
        self.button_option_2.setAutoFillBackground(False)
        self.button_option_2.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
                                           "color: rgb(255, 255, 255);\n"
                                           "/*background-color:pink;\n"
                                           "color: white;\n"
                                           "text-align: center;\n"
                                           "border-radius: 19px;  /* Butonu daire şeklinde yapmak için */\n"
                                           "width: 200px;  /* Butonun genişliği */\n"
                                           "height: 100px;  /* Butonun yüksekliği */\n"
                                           "border: 5px solid white; /* 5px kalınlığında mavi kenarlık */\n"
                                           "border-radius: 10px;      /* Kenarlıkları yuvarlatır (isteğe bağlı) */\n"
                                           "\n"
                                           "\n"
                                           "\n"
                                           "")
        self.button_option_2.setCheckable(False)
        self.button_option_2.setAutoExclusive(False)
        self.button_option_2.setDefault(False)
        self.button_option_2.setObjectName("button_option_2")
        self.button_option_1 = QtWidgets.QPushButton(self.widget)
        self.button_option_1.setGeometry(QtCore.QRect(800, 570, 271, 71))
        self.button_option_1.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
                                           "color: rgb(255, 255, 255);\n"
                                           "/*background-color:pink;\n"
                                           "color: white;\n"
                                           "text-align: center;\n"
                                           "border-radius: 19px;  /* Butonu daire şeklinde yapmak için */\n"
                                           "width: 200px;  /* Butonun genişliği */\n"
                                           "height: 100px;  /* Butonun yüksekliği */\n"
                                           "border: 5px solid white; /* 5px kalınlığında mavi kenarlık */\n"
                                           "border-radius: 10px;      /* Kenarlıkları yuvarlatır (isteğe bağlı) */\n"
                                           "\n"
                                           "\n"
                                           "\n"
                                           "")
        self.button_option_1.setObjectName("button_option_1")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(40, 60, 1111, 171))
        self.label.setStyleSheet("font: 42pt \"Goudy Old Style\";\n"
                                 "color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.frame = QtWidgets.QFrame(self.widget)
        self.frame.setGeometry(QtCore.QRect(510, 20, 581, 541))
        self.frame.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
                                 "\n"
                                 "border-color: rgb(255, 255, 255);\n"
                                 "color: white;\n"
                                 "text-align: center;\n"
                                 "/*border-radius: 80px;  /* Butonu daire şeklinde yapmak için */\n"
                                 "width: 200px;  /* Butonun genişliği */\n"
                                 "height: 100px;  /* Butonun yüksekliği */\n"
                                 "border: 5px solid white; /* 5px kalınlığında mavi kenarlık */\n"
                                 "border-radius: 10px;      /* Kenarlıkları yuvarlatır (isteğe bağlı) */\n"
                                 "\n"
                                 "\n"
                                 "\n"
                                 "\n"
                                 "\n"
                                 "")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.button_add = QtWidgets.QPushButton(self.frame)
        self.button_add.setGeometry(QtCore.QRect(410, 490, 141, 41))
        self.button_add.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
                                      "/*background-color:pink;\n"
                                      "color: white;\n"
                                      "text-align: center;\n"
                                      "/*border-radius: 19px;  /* Butonu daire şeklinde yapmak için */\n"
                                      "width: 200px;  /* Butonun genişliği */\n"
                                      "height: 100px;  /* Butonun yüksekliği */\n"
                                      "\n"
                                      "\n"
                                      "\n"
                                      "")
        self.button_add.setObjectName("button_add")
        self.drop_label = QtWidgets.QLabel(self.frame)
        self.drop_label.setGeometry(QtCore.QRect(40, 10, 512, 472))
        self.drop_label.setStyleSheet("border:none;\n"
                                      "")
        self.drop_label.setText("")
        self.drop_label.setObjectName("drop_label")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(10, 180, 491, 101))
        self.label_4.setStyleSheet("font: 42pt \"Goudy Old Style\";\n"
                                   "color: rgb(255, 255, 255);")
        self.label_4.setObjectName("label_4")
        self.ok_button = QtWidgets.QPushButton(self.widget)
        self.ok_button.setGeometry(QtCore.QRect(740, 670, 141, 61))
        self.ok_button.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
                                     "color: rgb(255, 255, 255);\n"
                                     "/*background-color:pink;\n"
                                     "color: white;\n"
                                     "text-align: center;\n"
                                     "border-radius: 19px;  /* Butonu daire şeklinde yapmak için */\n"
                                     "width: 200px;  /* Butonun genişliği */\n"
                                     "height: 100px;  /* Butonun yüksekliği */\n"
                                     "border: 5px solid white; /* 5px kalınlığında mavi kenarlık */\n"
                                     "border-radius: 10px;      /* Kenarlıkları yuvarlatır (isteğe bağlı) */\n"
                                     "\n"
                                     "\n"
                                     "\n"
                                     "")
        self.ok_button.setText("Analyze")
        self.ok_button.setObjectName("ok_button")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(770, 680, 91, 41))
        self.label_3.setStyleSheet("")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(-100, 100, 581, 971))
        self.label_2.setStyleSheet(
            "border-color: qlineargradient(spread:pad, x1:0.184, y1:0.869, x2:1, y2:0, stop:0 rgba(255, 170, 255, 255), stop:1 rgba(255, 255, 255, 255));\n"
            "\n"
            "image: url(:/clippart/breast-5472788_1280.png);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_2.raise_()
        self.frame.raise_()
        self.label.raise_()
        self.button_option_1.raise_()
        self.ok_button.raise_()
        self.label_3.raise_()
        self.button_option_2.raise_()
        self.label_4.raise_()

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.button_option_2.setText(_translate("Dialog", "Machine Learning"))
        self.button_option_1.setText(_translate("Dialog", "Deep Learning"))
        self.label.setText(_translate("Dialog", "Breast Cancer "))
        self.button_add.setText(_translate("Dialog", "Add"))
        self.label_4.setText(_translate("Dialog", "Detection System"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
