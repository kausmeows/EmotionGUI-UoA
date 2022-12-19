# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'visualize.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Visualise(object):
    def setupUi(self, Visualise):
        Visualise.setObjectName("Visualise")
        Visualise.resize(1440, 847)
        font = QtGui.QFont()
        font.setFamily("Academy Engraved LET")
        font.setPointSize(18)
        Visualise.setFont(font)
        self.label = QtWidgets.QLabel(Visualise)
        self.label.setGeometry(QtCore.QRect(10, 30, 611, 161))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Visualise)
        self.label_2.setGeometry(QtCore.QRect(10, 210, 221, 31))
        self.label_2.setObjectName("label_2")
        self.textEdit = QtWidgets.QTextEdit(Visualise)
        self.textEdit.setGeometry(QtCore.QRect(10, 240, 571, 31))
        self.textEdit.setObjectName("textEdit")
        self.pushButton = QtWidgets.QPushButton(Visualise)
        self.pushButton.setGeometry(QtCore.QRect(10, 280, 221, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Visualise)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 281, 261, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Visualise)
        self.pushButton_3.setGeometry(QtCore.QRect(240, 280, 71, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_5 = QtWidgets.QPushButton(Visualise)
        self.pushButton_5.setGeometry(QtCore.QRect(240, 430, 261, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(Visualise)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 430, 221, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_3 = QtWidgets.QLabel(Visualise)
        self.label_3.setGeometry(QtCore.QRect(10, 360, 261, 31))
        self.label_3.setObjectName("label_3")
        self.textEdit_2 = QtWidgets.QTextEdit(Visualise)
        self.textEdit_2.setGeometry(QtCore.QRect(10, 390, 571, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.pushButton_4 = QtWidgets.QPushButton(Visualise)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 480, 221, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_4 = QtWidgets.QLabel(Visualise)
        self.label_4.setGeometry(QtCore.QRect(10, 560, 381, 31))
        self.label_4.setObjectName("label_4")
        self.textEdit_3 = QtWidgets.QTextEdit(Visualise)
        self.textEdit_3.setGeometry(QtCore.QRect(10, 590, 571, 31))
        self.textEdit_3.setObjectName("textEdit_3")
        self.label_5 = QtWidgets.QLabel(Visualise)
        self.label_5.setGeometry(QtCore.QRect(10, 650, 221, 31))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Visualise)
        self.label_6.setGeometry(QtCore.QRect(10, 710, 71, 31))
        self.label_6.setObjectName("label_6")
        self.textEdit_4 = QtWidgets.QTextEdit(Visualise)
        self.textEdit_4.setGeometry(QtCore.QRect(80, 650, 221, 31))
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_5 = QtWidgets.QTextEdit(Visualise)
        self.textEdit_5.setGeometry(QtCore.QRect(80, 710, 221, 31))
        self.textEdit_5.setObjectName("textEdit_5")
        self.pushButton_7 = QtWidgets.QPushButton(Visualise)
        self.pushButton_7.setGeometry(QtCore.QRect(340, 700, 71, 41))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(Visualise)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 780, 341, 51))
        self.pushButton_8.setObjectName("pushButton_8")

        self.retranslateUi(Visualise)
        QtCore.QMetaObject.connectSlotsByName(Visualise)

    def retranslateUi(self, Visualise):
        _translate = QtCore.QCoreApplication.translate
        Visualise.setWindowTitle(_translate("Visualise", "Visualisation Screen"))
        self.label.setText(_translate("Visualise", "There are three different ways that this GUI allows you to plot emotional \n"
"coordinates on the model. To plot a CSV or WAV file, click the button to select \n"
"the corresponding file type and then click the plot button of the \n"
"corresponding file type. To plot manually, type appropriate values in the text \n"
"fields next to \'Valence\' and \'Arousal\' and then click plot.\n"
""))
        self.label_2.setText(_translate("Visualise", "1. Plot a CSV file:"))
        self.pushButton.setText(_translate("Visualise", "Select CSV File"))
        self.pushButton_2.setText(_translate("Visualise", "Plot Emotional Data of CSV File"))
        self.pushButton_3.setText(_translate("Visualise", "!"))
        self.pushButton_5.setText(_translate("Visualise", "Plot Emotional Data of CSV File"))
        self.pushButton_6.setText(_translate("Visualise", "Select CSV File"))
        self.label_3.setText(_translate("Visualise", "2. Plot a WAV file with a model:"))
        self.pushButton_4.setText(_translate("Visualise", "Change Model"))
        self.label_4.setText(_translate("Visualise", "3. Set coordinates and plot manually"))
        self.label_5.setText(_translate("Visualise", "Valence:"))
        self.label_6.setText(_translate("Visualise", "Arousal:"))
        self.pushButton_7.setText(_translate("Visualise", "Plot"))
        self.pushButton_8.setText(_translate("Visualise", "Main Menu"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Visualise = QtWidgets.QWidget()
    ui = Ui_Visualise()
    ui.setupUi(Visualise)
    Visualise.show()
    sys.exit(app.exec_())

