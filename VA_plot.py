import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QLinearGradient, QColor, QPalette, QBrush
from PyQt5 import QtGui
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtGui import QIcon
import numpy as np
import os
import utils.realtime_spectogram as rs
from spectogram import SpectrogramWidget
import runpy
from file_processing import FileProcessing
from utils.multimedia import VideoWindow
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random


class Valence_Arousal(QDialog):
    def __init__(self, parent=None):
        super(Valence_Arousal, self).__init__(parent)
        loadUi('ui/visualize.ui', self)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        # set the layout
        layout = QVBoxLayout()
        layout.setContentsMargins(750, 50, 50, 121)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)

        # V-A plot basics
        landmarkEmotions = ['angry', 'afraid', 'sad', 'bored', 'excited', 'interested', 'happy', 'pleased', 'relaxed', 'content']
        landmarkValence = (-0.7, -0.65, -0.8, -0.1, 0.37, 0.2, 0.5, 0.35, 0.6, 0.5)
        landmarkArousal = (0.65, 0.5, -0.15, -0.45, 0.9, 0.7, 0.5, 0.35, -0.3, -0.45)

        startR = (23, 253, 255, 137)
        startG = (255, 231, 146, 227)
        startB = (101, 45, 0, 181)

        endR = (251, 153, 9, 234)
        endG = (20, 34, 18, 115)
        endB = (20, 195, 121, 141)

    def plot(self):
        ''' plot some random stuff '''
        # random data
        # data = [random.random() for i in range(10)]

        # # instead of ax.hold(False)
        # self.figure.clear()

        # # create an axis
        # ax = self.figure.add_subplot(111)

        # # discards the old graph
        # # ax.hold(False) # deprecated, see above

        # # plot data
        # ax.plot(data, '*-')

        # # refresh canvas
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Valence_Arousal()
    main.show()

    sys.exit(app.exec_())
