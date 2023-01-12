from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QWidget, QPushButton

class FileProcessing(QtWidgets.QPushButton):
    def __init__(self):
        super(FileProcessing, self).__init__()

    def open_dialog_box(self) -> str:
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        return path
    
    
