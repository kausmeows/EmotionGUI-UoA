import sys
import csv
from PyQt5.uic import loadUi
from PyQt5 import QtGui
import sip
import matplotlib
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QDir, Qt, QUrl, QPoint, QTime, QProcess
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QLinearGradient, QColor, QPalette, QBrush
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtGui import QIcon
import numpy as np
from spectogram import SpectrogramWidget
from scipy.io import wavfile
from utils.file_processing import FileProcessing
from multimedia import VideoWindow

from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow, QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import Cursor

FS = 44100  # Hz
CHUNKSZ = 1024  # samples


class MainWindow(QMainWindow, QPushButton):
    def __init__(self):
        super(MainWindow, self).__init__()

        loadUi('ui/home.ui', self)
        pic = QPixmap('static/uoa_logo.png')
        self.imglabel.setPixmap(pic)

        self.visualiseB = self.findChild(
            QtWidgets.QPushButton, 'visualise_button')
        self.visualiseB.clicked.connect(self.goto_visualize)

        self.annotateB = self.findChild(
            QtWidgets.QPushButton, 'annotate_button')
        self.annotateB.clicked.connect(self.goto_annotate)

        self.liveAudioB = self.findChild(
            QtWidgets.QPushButton, 'liveAudio_button')
        self.liveAudioB.clicked.connect(self.goto_liveAudio)

    def goto_visualize(self):
        visualise = visualisationScreen()
        widget.addWidget(visualise)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def goto_annotate(self):
        annotate = annotationScreen()
        widget.addWidget(annotate)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def goto_liveAudio(self):
        # os.system('python spectogram.py')
        # liveAudio = SpectrogramWidget()
        # widget.addWidget(liveAudio)
        # widget.setCurrentIndex(widget.currentIndex() + 1)
        # liveAudioScreen(liveAudio.getHomeButton())

        liveAudio = liveAudioScreen()
        widget.addWidget(liveAudio)
        widget.setCurrentIndex(widget.currentIndex() + 1)


class visualisationScreen(QWidget):
    def __init__(self):
        super(visualisationScreen, self).__init__()
        loadUi('ui/visualize.ui', self)

        # -------------------------------------------------------------
        # to get the dialog box when the 'Select CSV' button is clicked and write the path of file on the text editor
        self.select_csv = self.findChild(
            QtWidgets.QPushButton, 'select_csv')
        self.select_csv.clicked.connect(
            lambda: self.textEdit.setText(FileProcessing.open_dialog_box(self)))
        # --------------------------------------------------------------

        # -------------------------------------------------------------
        # to get the dialog box when the 'Select WAV' button is clicked and write the path of file on the text editor
        self.select_wav = self.findChild(
            QtWidgets.QPushButton, 'select_wav')
        self.select_wav.clicked.connect(
            lambda: self.textEdit_2.setText(FileProcessing.open_dialog_box(self)))
        # --------------------------------------------------------------

        # -------------------------------------------------------------
        # to plot the csv file's points into the VA plot
        self.plot_VA = self.findChild(
            QtWidgets.QPushButton, 'plot_VA')
        self.plot_VA.clicked.connect(self.updateCircle)
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        self.plotButton_manual = self.findChild(
            QtWidgets.QPushButton, 'plotButton_manual')
        self.plotButton_manual.clicked.connect(self.plotVA_Manual)
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # to clear the plot
        self.clear_button = self.findChild(
            QtWidgets.QPushButton, 'clear_button')
        self.clear_button.clicked.connect(self.clear_plot)
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        self.homeB = self.findChild(
            QtWidgets.QPushButton, 'pushButton_8')
        self.homeB.clicked.connect(self.goto_home)
        # -------------------------------------------------------------

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
        self.axes = self.figure.subplots()

        theta = np.linspace(0, 2*np.pi, 100)
        radius = 1

        a = radius*np.cos(theta)
        b = radius*np.sin(theta)

        self.axes.plot(a, b, color='k', linewidth=1)
        self.axes.set_aspect(1)

        self.axes.hlines(y=0, xmin=-1, xmax=1, linewidth=0.7, color='k')
        self.axes.vlines(x=0, ymin=-1, ymax=1, linewidth=0.7, color='k')

        self.axes.tick_params(axis='both', which='major', labelsize=5)
        self.axes.tick_params(axis='both', which='minor', labelsize=5)

        # V-A plot basic landmark emotions coordinates
        self.landmarkEmotions = ['angry', 'afraid', 'sad', 'bored', 'excited',
                                 'interested', 'happy', 'pleased', 'relaxed', 'content']
        self.landmarkValence = (-0.7, -0.65, -0.8, -0.1, 0.37,
                                0.2, 0.5, 0.35, 0.6, 0.5)
        self.landmarkArousal = (0.65, 0.5, -0.15, -0.45, 0.9,
                                0.7, 0.5, 0.35, -0.3, -0.45)

        for point in range(len(self.landmarkEmotions)):
            self.axes.text(self.landmarkValence[point], self.landmarkArousal[point],
                           self.landmarkEmotions[point], fontstyle='italic', fontsize='xx-small')

        self.axes.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
        self.axes.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)

        self.canvas.draw()

    def updateCircle(self):
        self.RGB_values = [[255 / 255, 0 / 255, 0 / 255], [255 / 255, 255 / 255, 0 / 255],
                           [0 / 255, 0 / 255, 255 / 255]]  # start with red, yellow and blue
        csv_address = self.textEdit.toPlainText()

        if(csv_address != ''):
            # Get the csv file address from the text edit
            VA = []
            with open(csv_address, 'r') as file:
                csvreader = csv.reader(file)
                header = next(csvreader)
                for row in csvreader:
                    VA.append(row)
            # print(VA)

        self.last_time_sec = VA[len(VA)-1][0]
        for VA_point in range(len(VA)):
            time = VA[VA_point][0]
            valence = VA[VA_point][1]
            arousal = VA[VA_point][2]
            if(float(valence) >= -1 and float(valence) <= 1 and float(arousal) >= -1 and float(arousal) <= 1):
                self.plotColorGradedPoints(valence, arousal, time)

        self.canvas.draw()

    def plotColorGradedPoints(self, valence, arousal, time):
        if(float(time) <= float(self.last_time_sec) / 3):
            self.axes.scatter(float(valence), float(arousal),
                              color=self.RGB_values[0], s=5)
            # self.RGB_values[0][0] = self.RGB_values[0][0] - 0.001
            self.RGB_values[0][1] = self.RGB_values[0][1] + 0.05
            # self.RGB_values[0][2] = self.RGB_values[0][2] + 0.01

        if(float(time) > float(self.last_time_sec) / 3 and float(time) <= 2 * float(self.last_time_sec) / 3):
            self.axes.scatter(float(valence), float(arousal),
                              color=self.RGB_values[1], s=5)
            self.RGB_values[1][0] = self.RGB_values[1][0] - 0.02
            self.RGB_values[1][1] = self.RGB_values[1][1] - 0.02
            self.RGB_values[1][2] = self.RGB_values[1][2] + 0.05

        elif(float(time) > 2 * float(self.last_time_sec) / 3):
            self.axes.scatter(float(valence), float(
                arousal), color=self.RGB_values[2], s=5)
            # self.RGB_values[2][0] = self.RGB_values[2][0] + 0.05
            # self.RGB_values[2][1] = self.RGB_values[2][1] + 0.05
            self.RGB_values[2][2] = self.RGB_values[2][2] - 0.05

    def plotVA_Manual(self):
        manual_valence = self.valence_field.toPlainText()
        manual_arousal = self.arousal_field.toPlainText()

        if(manual_valence != '' and manual_arousal != '' and float(manual_valence) >= -1 and float(manual_valence) <= 1
                and float(manual_arousal) >= -1 and float(manual_arousal) <= 1):
            self.axes.scatter(float(manual_valence), float(
                manual_arousal), color='blue', s=5)
        self.canvas.draw()

    def clear_plot(self):
        self.createCircle()

    def goto_home(self):
        home = MainWindow()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex() - 1)
        widget.setCurrentWidget(home)


class annotationScreen(QMainWindow):
    def __init__(self):
        super(annotationScreen, self).__init__()
        loadUi('ui/annotate.ui', self)

        self.setWindowTitle("Video Player")

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videoWidget = QVideoWidget()

        # create open button
        openBtn = QPushButton('Open Video/Audio File')
        openBtn.clicked.connect(self.openFile)

        # to clear the plot
        self.clear_button = self.findChild(
            QtWidgets.QPushButton, 'clear_button')
        self.clear_button.clicked.connect(self.clear_plot)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        controlLayout = self.horizontalLayout
        controlLayout.addWidget(openBtn)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = self.multimedia
        layout.addWidget(videoWidget)
        layout.addWidget(self.errorLabel)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

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

        self.figure.canvas.mpl_connect(
            'button_press_event', self.annotateOnClick)

        # set the layout
        va_layout = self.va_plot
        va_layout.addWidget(self.toolbar)
        va_layout.addWidget(self.canvas)

        # -------------------------------------------------------------
        # audio visualizer
        self.open_audio = self.findChild(
            QtWidgets.QPushButton, 'open_audio')
        # self.open_audio.clicked.connect(self.audioVis)
        # if(self.audio_filepath != ''):
        #     self.audioVis()

        self.clear_audio_vis = self.findChild(
            QtWidgets.QPushButton, 'clear_audio_vis')
        # self.clear_audio_vis.clicked.connect(self.clearAudioVisualizer)
        # -------------------------------------------------------------

        self.homeB = self.findChild(
            QtWidgets.QPushButton, 'home_button_annotator')
        self.homeB.clicked.connect(self.goto_home)

    def audioVis(self, filepath):
        # self.clearAudioVisualizer(self.audio_figure)
        # self.clearAudioVisualizer(self.audio_canvas)
        # self.clearAudioVisualizer(self.audio_toolbar)

        # a figure instance to plot on
        self.audio_figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.audio_canvas = FigureCanvas(self.audio_figure)

        # this is the Navigation widget
        self.audio_toolbar = NavigationToolbar(self.audio_canvas, self)

        axes = self.audio_figure.subplots()

        samplingFrequency, signalData = wavfile.read(filepath)
        # Plot the signal read from wav file
        plt.title('Spectrogram', fontsize=3)
        # plt.rcParams["figure.figsize"] = [10, 10]
        plt.subplots_adjust(left=0.100, right=0.900, top=0.860, bottom=0.140)
        axes.tick_params(axis='both', which='major', labelsize=3)
        axes.tick_params(axis='both', which='minor', labelsize=1)
        plt.specgram(signalData, Fs=samplingFrequency)
        plt.xlabel('Time')
        plt.ylabel

        self.audio_layout = self.audio_visualizer
        self.audio_layout.addWidget(self.audio_toolbar)
        self.audio_layout.addWidget(self.audio_canvas)

        # vis = AudioVisualizer('hello_UoA.wav')
        # p1 = Process(target=vis.playing_audio, args=())
        # p1.start()
        # p2 = Process(target=vis.showing_audiotrack, args=())
        # p2.start()
        # p1.join()
        # p2.join()

    def clearAudioVisualizer(self, widgetName):
        self.audio_layout.removeWidget(widgetName)
        widgetName.deleteLater()
        widgetName = None

    def createCircle(self):
        self.figure.clear()
        self.axes = self.figure.subplots()

        theta = np.linspace(0, 2*np.pi, 100)
        radius = 1

        a = radius*np.cos(theta)
        b = radius*np.sin(theta)

        self.axes.plot(a, b, color='k', linewidth=1)
        self.axes.set_aspect(1)

        self.axes.hlines(y=0, xmin=-1, xmax=1, linewidth=0.7, color='k')
        self.axes.vlines(x=0, ymin=-1, ymax=1, linewidth=0.7, color='k')

        self.axes.tick_params(axis='both', which='major', labelsize=5)
        self.axes.tick_params(axis='both', which='minor', labelsize=5)

        # V-A plot basic landmark emotions coordinates
        self.landmarkEmotions = ['angry', 'afraid', 'sad', 'bored', 'excited',
                                 'interested', 'happy', 'pleased', 'relaxed', 'content']
        self.landmarkValence = (-0.7, -0.65, -0.8, -0.1, 0.37,
                                0.2, 0.5, 0.35, 0.6, 0.5)
        self.landmarkArousal = (0.65, 0.5, -0.15, -0.45, 0.9,
                                0.7, 0.5, 0.35, -0.3, -0.45)

        for point in range(len(self.landmarkEmotions)):
            self.axes.text(self.landmarkValence[point], self.landmarkArousal[point],
                           self.landmarkEmotions[point], fontstyle='italic', fontsize='xx-small')

        self.axes.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
        self.axes.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)

        self.canvas.draw()

    def annotateOnClick(self, event):
        print(round(event.xdata, 2), round(event.ydata, 2))
        if(event.xdata >= -1 and event.xdata <= 1 and event.ydata >= -1 and event.ydata <= 1):
            self.axes.scatter(round(event.xdata, 2), round(
                event.ydata, 2), color='red', s=5)
            self.canvas.draw()
            self.saveAsCSV(event.xdata, event.ydata)
    
    def saveAsCSV(self, xdata, ydata):
        pass

    def clear_plot(self):
        self.createCircle()

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Movie/Audio", QDir.homePath())

        if fileName != '':
            self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.audioVis(fileName)

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)
        mtime = QTime(0, 0, 0, 0)
        mtime = mtime.addMSecs(self.mediaPlayer.position())
        self.lbl.setText(mtime.toString())

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)
        mtime = QTime(0, 0, 0, 0)
        mtime = mtime.addMSecs(self.mediaPlayer.duration())
        self.elbl.setText(mtime.toString())

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

    def goto_home(self):
        home = MainWindow()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex() - 1)
        widget.setCurrentWidget(home)


class liveAudioScreen(QMainWindow, QWidget):
    def __init__(self):
        super(liveAudioScreen, self).__init__()
        # loadUi('ui/audio.ui', self)
        w = SpectrogramWidget()
        w.read_collected.connect(w.update)

        # super(liveAudioScreen, self).__init__()
        # self.homeButton = homeB
        # self.homeButton.clicked.connect(self.goto_home)

    def goto_home(self):
        home = MainWindow()
        widget.addWidget(home)
        widget.setCurrentIndex(widget.currentIndex() - 1)
        widget.setCurrentWidget(home)


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        widget = QtWidgets.QStackedWidget()

        home = MainWindow()

        widget.addWidget(home)
        widget.showMaximized()
        sys.exit(app.exec_())
    except:
        print("Exit")
