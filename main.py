import sys
import csv
import wave
from PyQt5.uic import loadUi
import pyaudio
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import QDir, Qt, QUrl, QPoint, QTime, QProcess
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QLinearGradient, QColor, QPalette, QBrush
from PyQt5.QtCore import QDir, Qt, QUrl, QTimer
from PyQt5.QtGui import QIcon
import numpy as np
from spectogram import MicrophoneRecorder, SpectrogramWidget
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
import pyqtgraph as pg
from utils import audio_processing as AP

FS = 44100  # Hz
CHUNKSZ = 1024  # samples
form_class = uic.loadUiType("ui/audio.ui")[0]
counter = 0


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
            self.axes.scatter(self.landmarkValence,
                              self.landmarkArousal, color='k', s=5)
            self.axes.text(self.landmarkValence[point] + 0.02, self.landmarkArousal[point] + 0.02,
                           self.landmarkEmotions[point], fontsize='xx-small')

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
            self.RGB_values[2][2] = self.RGB_values[2][2] - 0.02

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

        self.valence_points = []
        self.arousal_points = []
        self.time_points = []
        self.count_out_of_bounds = 0

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

        self.save_CSV_button.clicked.connect(self.saveAsCSV)

        self.homeB = self.findChild(
            QtWidgets.QPushButton, 'home_button_annotator')
        self.homeB.clicked.connect(self.goto_home)

        # to keep calling the positionChanged function in order to update the time and store it
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.positionChanged)
        self.timer.start(10)

    def audioVis(self, filepath):
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
            self.axes.scatter(self.landmarkValence,
                              self.landmarkArousal, color='k', s=5)
            self.axes.text(self.landmarkValence[point] + 0.02, self.landmarkArousal[point] + 0.02,
                           self.landmarkEmotions[point], fontsize='xx-small')

        self.axes.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
        self.axes.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)

        self.canvas.draw()

    def annotateOnClick(self, event):
        print(round(event.xdata, 2), round(event.ydata, 2))
        if(event.xdata >= -1 and event.xdata <= 1 and event.ydata >= -1 and event.ydata <= 1):
            self.axes.scatter(round(event.xdata, 2), round(
                event.ydata, 2), color='red', s=5)
            self.canvas.draw()
            self.savePoints(round(event.xdata, 2), round(event.ydata, 2))
        else:
            self.count_out_of_bounds += 1
            self.out_of_bounds_lbl.setText(
                "You have clicked out of the annotation model {} times".format(self.count_out_of_bounds))

    def savePoints(self, xdata, ydata):
        self.valence_points.append(xdata)
        self.arousal_points.append(ydata)
        self.time_points.append(self.seconds)

    def saveAsCSV(self):
        header = ["Time", "Valence", "Arousal"]
        with open('csv_outputs/annotation/example.csv', 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            rows = [(str(time), str(valence), str(arousal)) for time, valence, arousal in zip(self.time_points,
                                                                                              self.valence_points, self.arousal_points)]
            writer.writerows(rows)

    def clear_plot(self):
        self.createCircle()
        self.out_of_bounds_lbl.clear()

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


    def positionChanged(self):
        position = self.mediaPlayer.position()
        self.positionSlider.setValue(position)

        mtime = QTime(0, 0, 0, 0)
        mtime = mtime.addMSecs(position)
        formatted_time = mtime.toString("HH:mm:ss")
        self.lbl.setText(formatted_time)

        seconds = mtime.second()
        milliseconds = mtime.msec()
        time_in_seconds = seconds + (milliseconds/1000)
        self.seconds = "{:.2f}".format(time_in_seconds) # this gives seconds upto two decimal places


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
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(liveAudioScreen, self).__init__()
        loadUi('ui/audio.ui', self)

        self.timer.setText('0 sec')
        self.stop_pressed = False
        self.progressBar.setValue(0)

        self.mic = MicrophoneRecorder(self.read_collected)
        self.read_collected.connect(self.update)

        # time (seconds) between reads
        interval = FS/CHUNKSZ
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.mic.read)
        self.t.start(1000/interval)  # QTimer takes ms

        self.timerrr = QtCore.QTimer()
        self.timerrr.timeout.connect(self.update_counter)
        self.timerrr.start(1000)  # QTimer takes ms

        self.img = pg.ImageItem()
        self.graphicsView.addItem(self.img)
        self.graphicsView.setTitle('Real-Time Spectogram')
        # self.graphicsView.setYRange(0, 8000)
        self.graphicsView.hideAxis('bottom')
        self.graphicsView.hideAxis('left')
        # self.graphicsView.setLimits(yMin=0, yMax=8000)
        self.graphicsView_wave.setTitle('Audio Waveform')
        self.graphicsView_wave.setLabel('left', 'Amplitude')
        self.graphicsView_wave.hideAxis('bottom')

        self.audiosources, self.audiosourceIDs, self.PyAudioObject = AP.listaudiodevices()

        for source in self.audiosources:
            self.data_source.addItem(source)

        self.img_array = np.zeros((1000, int(CHUNKSZ/2+1)))

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [
                         0, 0, 0, 255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        # self.img.setLevels([-10, 600])
        self.img.setLevels([-50, 600])

        # plotting the audio waveform
        self.pdataitem = self.graphicsView_wave.plot(self.mic.frames)

        self.stop = self.findChild(
            QtWidgets.QPushButton, 'stop')
        self.stop.clicked.connect(self.stop_recording)

        self.save = self.findChild(
            QtWidgets.QPushButton, 'save_wav')
        self.save.clicked.connect(self.save_audio)

        self.save_spec = self.findChild(
            QtWidgets.QPushButton, 'save_png')
        self.save_spec.clicked.connect(self.save_photo)

        # prepare window for later use
        self.win = np.hanning(CHUNKSZ)
        self.show()

    def stop_recording(self):
        self.mic.stream.stop_stream()
        self.mic.stream.close()
        self.mic.p.terminate()

        self.stop_pressed = True

    def update(self, chunk):
        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(chunk*self.win) / CHUNKSZ
        # get magnitude
        psd = abs(spec)
        # convert to dB scale
        psd = 20 * np.log10(psd)

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = psd

        # update the plotted variable for audio waveform
        self.pdataitem.setData(chunk)

        self.img.setImage(self.img_array, autoLevels=False)

    def save_audio(self):
        sound_file = wave.open("hello_UoA.wav", "wb")
        print("saving")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.mic.p.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(FS)
        sound_file.writeframes(b''.join(self.mic.frames))
        sound_file.close()

    def save_photo(self):
        # Read the wav file (mono)
        samplingFrequency, signalData = wavfile.read(
            'hello_UoA.wav')
        # Plot the signal read from wav file
        plt.subplot(211)
        plt.title('Spectrogram')
        plt.plot(signalData)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplot(212)
        plt.specgram(signalData, Fs=samplingFrequency)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    def update_counter(self):
        global counter
        if self.stop_pressed != True:
            counter += 1
            self.timer.clear()
            self.progressBar.setValue(counter)
            self.timer.setText(str(counter) + ' sec')

        else:
            return

        self.main_menu.clicked.connect(self.goto_home)

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
