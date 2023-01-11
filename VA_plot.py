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

        # draw the circle
        self.createCircle()

        # set the layout
        layout = QVBoxLayout()
        layout.setContentsMargins(750, 50, 50, 121)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # layout.addWidget(self.button)
        self.setLayout(layout)

    def createCircle(self):
        self.figure.clear()
        axes = self.figure.subplots()

        theta = np.linspace(0, 2*np.pi, 100)
        radius = 1

        a = radius*np.cos(theta)
        b = radius*np.sin(theta)

        axes.plot(a, b, color='k', linewidth=1)
        axes.set_aspect(1)

        axes.hlines(y=0, xmin=-1, xmax=1, linewidth=0.7, color='k')
        axes.vlines(x=0, ymin=-1, ymax=1, linewidth=0.7, color='k')

        axes.tick_params(axis='both', which='major', labelsize=5)
        axes.tick_params(axis='both', which='minor', labelsize=5)

        # V-A plot basics
        landmarkEmotions = ['angry', 'afraid', 'sad', 'bored', 'excited',
                            'interested', 'happy', 'pleased', 'relaxed', 'content']
        landmarkValence = (-0.7, -0.65, -0.8, -0.1, 0.37,
                           0.2, 0.5, 0.35, 0.6, 0.5)
        landmarkArousal = (0.65, 0.5, -0.15, -0.45, 0.9,
                           0.7, 0.5, 0.35, -0.3, -0.45)

        for point in range(len(landmarkEmotions)):
            axes.text(landmarkValence[point], landmarkArousal[point],
                      landmarkEmotions[point], fontsize='xx-small')

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Valence_Arousal()
    main.show()

    sys.exit(app.exec_())
