# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'resultscreen.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
import test_rc

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(0, 0, 1200, 800))
        self.widget.setStyleSheet("QWidget#widget{\n"
                                  "\n"
                                  "border-image: url(:/clippart/78505c683664dc60f3b10524f15de2eb.jpg);\n"
                                  "}\n"
                                  "")
        self.widget.setObjectName("widget")
        self.frame_results = QtWidgets.QFrame(self.widget)
        self.frame_results.setGeometry(QtCore.QRect(90, 110, 1051, 500))
        self.frame_results.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
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
        self.frame_results.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_results.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_results.setObjectName("frame_results")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(500, 30, 181, 61))
        self.label.setStyleSheet("font: 42pt \"Goudy Old Style\";\n"
                                 "color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.button_upload_page = QtWidgets.QPushButton(self.widget)
        self.button_upload_page.setGeometry(QtCore.QRect(90, 680, 271, 71))
        self.button_upload_page.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
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
        self.button_upload_page.setObjectName("button_upload_page")
        self.button_exit = QtWidgets.QPushButton(self.widget)
        self.button_exit.setGeometry(QtCore.QRect(830, 680, 271, 71))
        self.button_exit.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
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
        self.button_exit.setObjectName("button_exit")
        self.label_diagnosis = QtWidgets.QLabel(self.widget)
        self.label_diagnosis.setGeometry(QtCore.QRect(90, 620, 1031, 51))
        self.label_diagnosis.setStyleSheet("font: 75 14pt \"Goudy Old Style\";\n"
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
        self.label_diagnosis.setText("")
        self.label_diagnosis.setObjectName("label_diagnosis")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Result"))
        self.button_upload_page.setText(_translate("Dialog", "Upload Page"))
        self.button_exit.setText(_translate("Dialog", "Exit"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
