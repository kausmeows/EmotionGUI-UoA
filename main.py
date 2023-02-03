import random
import sys
import csv
import wave
from PyQt5.uic import loadUi
import pyaudio
import matplotlib.pyplot as plt
import librosa as lbr
from PyQt5 import QtWidgets, QtCore, uic
from pydub import AudioSegment, silence
from PyQt5.QtCore import QDir, Qt, QUrl, QTime
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QWidget, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir, Qt, QUrl, QTimer
from PyQt5.QtGui import QIcon
import numpy as np
from utils.spectogram import MicrophoneRecorder
from scipy.io import wavfile
from utils.file_processing import FileProcessing
import pandas as pd

from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QLabel,
							 QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow, QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pyqtgraph as pg
from utils import audio_processing as AP
import keras
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Audio

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

		self.manual_valence_vec = []
		self.manual_arousal_vec = []

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
		# to plot the wav file's points into the VA plot
		self.plot_predicted = self.findChild(
			QtWidgets.QPushButton, 'plot_predicted')
		self.plot_predicted.clicked.connect(self.plotPredictedEmotions)
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
		
		self.saveCSV_visualize = self.findChild(
			QtWidgets.QPushButton, 'saveCSV_visualize')
		self.saveCSV_visualize.clicked.connect(self.saveCSVPredicted)

		# -------------------------------------------------------------
		self.homeB = self.findChild(
			QtWidgets.QPushButton, 'pushButton_8')
		self.homeB.clicked.connect(self.goto_home)
		# -------------------------------------------------------------
  
		self.info_csv_btn.clicked.connect(self.openInfoBox)

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
  
	def openInfoBox(self):
		msg = QMessageBox()
		msg.setWindowTitle("Information")
		msg.setText("To plot a CSV file using this application, please make sure that the the first three columns have the titles 'Time', 'Valence' and 'Arousal' respectively. The coordinates of the points to be plotted are then below the respective columns.")
		msg.setIcon(QMessageBox.Information)
		msg.setStandardButtons(QMessageBox.Cancel)
		msg.exec_()

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

		self.axes.tick_params(axis='both', which='major', labelsize=10)
		self.axes.tick_params(axis='both', which='minor', labelsize=10)

		# to fit the matplotlib figure into the canvas
		plt.xlabel("Valence")
		plt.ylabel("Arousal")

		# V-A plot basic landmark emotions coordinates
		self.landmarkEmotions = ['angry', 'afraid', 'sad', 'bored', 'excited',
								 'interested', 'happy', 'pleased', 'relaxed', 'content']
		self.landmarkValence = (-0.7, -0.65, -0.8, -0.1, 0.37,
								0.2, 0.5, 0.35, 0.6, 0.5)
		self.landmarkArousal = (0.65, 0.5, -0.15, -0.45, 0.9,
								0.7, 0.5, 0.35, -0.3, -0.45)

		for point in range(len(self.landmarkEmotions)):
			self.axes.scatter(self.landmarkValence,
							  self.landmarkArousal, color='k', s=15)
			self.axes.text(self.landmarkValence[point] + 0.02, self.landmarkArousal[point] + 0.02,
						   self.landmarkEmotions[point], fontsize='large')

		self.axes.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
		self.axes.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)

		self.canvas.draw()

	def updateCircle(self):
		self.RGB_values = [[1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, 1]]  # start with red, yellow and blue
		csv_address = self.textEdit.toPlainText()

		if (csv_address != ''):
			# Get the csv file address from the text edit
			VA = []
			with open(csv_address, 'r') as file:
				csvreader = csv.reader(file)
				header = next(csvreader)
				for row in csvreader:
					VA.append(row)
	
		self.axes.text(float(VA[0][1]) + 0.02, float(VA[0][2]) + 0.02, 'start', color='red', size='large')
		self.axes.text(float(VA[len(VA)-1][1]) + 0.02, float(VA[len(VA)-1][2]), 'end', color='blue', size='large')
  
		self.last_time_sec = VA[len(VA)-1][0]
		for VA_point in range(len(VA)):
			time = VA[VA_point][0]
			valence = VA[VA_point][1]
			arousal = VA[VA_point][2]
			if (float(valence) >= -1 and float(valence) <= 1 and float(arousal) >= -1 and float(arousal) <= 1):
				self.plotColorGradedPoints(valence, arousal, time)

		self.canvas.draw()

	# Preprocessing & Feature Extraction
	def silenceStampExtract(self, audioPath, length):
		myaudio = AudioSegment.from_wav(audioPath)
		slc = silence.detect_silence(
			myaudio, min_silence_len=1000, silence_thresh=-32)
		slc = [((start/1000), (stop/1000))
			   for start, stop in slc]  # convert to sec
		slc = np.array(
			[item for sublist in slc for item in sublist])  # flatten
		slc = np.around(slc, 2)  # keep 2 dp
		# evaluate points to nearest previous 40ms stamp
		slc = (slc*100-slc*100 % 4)/100
		# Tag filling
		tagList = list()
		slc = np.append(slc, 9999)  # use length to determine the end
		time = 0.00
		idx = 0
		if slc[0] == 0:
			# filling start with Stag = 'S'
			tag = 'S'
			idx += 1
		else:
			# filling start with Stag = 'V'
			tag = 'V'
		for i in range(length):
			if time >= slc[idx]:
				idx += 1
				tag = 'V' if (idx % 2 == 0) else 'S'
			else:
				pass
			tagList.append(tag)
			time += 0.02
		return pd.DataFrame(tagList, columns=['voiceTag'])

	def featureExtract(self, audioFile):
		# parameters of 20ms window under 44.1kHZ
		# samplingRate = 44100
		# frameLength = 882
		frameLengthT = 0.02  # 20ms
		mfccNum = 5

		x, sr = lbr.load(audioFile, sr=None, mono=True)
		frameLength = int(sr*frameLengthT)
		frames = range(len(x)//frameLength+1)
		t = lbr.frames_to_time(frames, sr=sr, hop_length=frameLength)

		################## Energy##################
		rms = ((lbr.feature.rms(x, frame_length=frameLength,
				hop_length=frameLength, center=True))[0])
		rms = 20*np.log10(rms)

		################## F0##################
		f0Result = lbr.yin(x, 50, 300, sr, frame_length=frameLength*4)

		################## MFCC##################
		# Transpose mfccResult matrix
		mfccResult = lbr.feature.mfcc(
			x, sr=sr, n_mfcc=mfccNum, hop_length=frameLength).T

		########################################
		dfT = pd.DataFrame(t, columns=['Time'])
		dfR = pd.DataFrame(rms, columns=['RMS'])
		dfF = pd.DataFrame(f0Result, columns=['F0'])

		# MFCC Title
		mfccTitle = list()
		for num in range(mfccNum):
			mfccTitle.append('MFCC'+str(num+1))
		dfM = pd.DataFrame(mfccResult, columns=mfccTitle)

		return dfT.join(dfR).join(dfF).join(dfM)

	def dataframes(self):
		self.currentDf = self.featureExtract(self.wav_address)
		self.tagDf = self.silenceStampExtract(
			self.wav_address, self.currentDf.shape[0])
		self.currentDf = self.currentDf.join(self.tagDf)

	# Machine Learning

	# prepare data for lstms
	def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
		n_vars = 1 if type(data) is list else data.shape[1]
		df = pd.DataFrame(data)
		cols, names = list(), list()
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(df.shift(-i))
			if i == 0:
				names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
			else:
				names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		# put it all together
		agg = pd.concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			agg.dropna(inplace=True)
		return agg

	def predict(self):
		# Define scaler, feature number and number of step looking back
		scale_range = (0, 1)
		scaler = MinMaxScaler(feature_range=scale_range)
		n_steps = 24  # exclude the current step
		n_features = 7

		transformTarget = True

		testingDataset = self.currentDf
		testingDataset = testingDataset[['RMS', 'F0',
										 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5']]

		# load and build testing dataset
		values = testingDataset.values
		# normalize features
		testingScaled = scaler.fit_transform(values)
		# frame as supervised learning
		reframed = self.series_to_supervised(testingScaled, n_steps, 1)
		print(reframed.shape)
		values = reframed.values
		test = values
		test_X = test
		# reshape input to be 3D [samples, timesteps (n_steps before + 1 current step), features]
		test_X = test_X.reshape((test_X.shape[0], n_steps + 1, n_features))

		arousalModelPath = '/Users/kaustubh/Desktop/EmotionGUI-UoA/models/saves/bi-lstm/mArousal.hdf5'
		valenceModelPath = '/Users/kaustubh/Desktop/EmotionGUI-UoA/models/saves/bi-lstm/mValence.hdf5'

		arousalModel = keras.models.load_model(arousalModelPath)
		valenceModel = keras.models.load_model(valenceModelPath)

		# make a prediction
		if transformTarget:
			inv_yPredict = arousalModel.predict(test_X)
			yPredict = arousalModel.predict(test_X)

		self.a_pred_test_list = [i for i in yPredict]

		# make a prediction
		if transformTarget:
			inv_yPredict = valenceModel.predict(test_X)
			yPredict = valenceModel.predict(test_X)

		self.v_pred_test_list = [i for i in yPredict]
		self.time_array = self.currentDf[['Time']].to_numpy()
		self.time_array = self.time_array[24:len(self.time_array)]

	def plotPredictedEmotions(self):
		self.RGB_values = [[255 / 255, 0 / 255, 0 / 255], [255 / 255, 255 / 255, 0 / 255],
						   [0 / 255, 0 / 255, 255 / 255]]  # start with red, yellow and blue
		self.wav_address = self.textEdit_2.toPlainText()
		Audio(self.wav_address)
		if (self.wav_address != ''):
			# Get the wav file address from the text edit
			self.dataframes()
			self.predict()

		self.last_time_sec = self.time_array[len(self.time_array) - 1]
		for VA_point in range(len(self.v_pred_test_list)):
			time = self.time_array[VA_point]
			valence = self.v_pred_test_list[VA_point]
			arousal = self.a_pred_test_list[VA_point]
			if (float(valence) >= -1 and float(valence) <= 1 and float(arousal) >= -1 and float(arousal) <= 1):
				self.plotColorGradedPoints(valence, arousal, time)	
		self.canvas.draw()
  
	def plotColorGradedPoints(self, valence, arousal, time):
		alpha = 1
		if (float(time) <= float(self.last_time_sec) / 3):
			alpha = 1 - float(time) / (float(self.last_time_sec) / 3)
			if alpha < 0:
					alpha = 0
			elif alpha > 1:
				alpha = 1
			self.axes.scatter(float(valence), float(arousal),
							  color=self.RGB_values[0], s=18, alpha=alpha)
		
		if (float(time) > float(self.last_time_sec) / 3 and float(time) <= 2 * float(self.last_time_sec) / 3):
			alpha = 1 - (float(time) - float(self.last_time_sec) / 3) / (float(self.last_time_sec) / 3)
			if alpha < 0:
					alpha = 0
			elif alpha > 1:
				alpha = 1
			self.axes.scatter(float(valence), float(arousal),
							  color=self.RGB_values[1], s=18, alpha=alpha)
		
		elif (float(time) > 2 * float(self.last_time_sec) / 3):
			alpha = 1 - (float(time) - 2 * float(self.last_time_sec) / 3) / (float(self.last_time_sec) / 3)
			if alpha < 0:
					alpha = 0
			elif alpha > 1:
				alpha = 1
			self.axes.scatter(float(valence), float(
				arousal), color=self.RGB_values[2], s=18, alpha=alpha)
		self.canvas.draw()

	def saveCSVPredicted(self):
		header = ["Time", "Valence", "Arousal"]

		# this opens a dialog box to save the file
		filename = 'visualized-file'
		filename, _ = QFileDialog.getSaveFileName(
			self, "Save CSV file", filename, "CSV Files (*.csv)"
		)
		if filename:
			with open(filename, 'w+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(header)
				rows = [(str(time[0]), str(valence[0]), str(arousal[0])) for time, valence, arousal in zip(self.time_array, 
																		self.v_pred_test_list, self.a_pred_test_list)]
				writer.writerows(rows)

	def plotVA_Manual(self):
		manual_valence = self.valence_field.toPlainText()
		manual_arousal = self.arousal_field.toPlainText()

		if (manual_valence != '' and manual_arousal != '' and float(manual_valence) >= -1 and float(manual_valence) <= 1
		   and float(manual_arousal) >= -1 and float(manual_arousal) <= 1):
			self.axes.scatter(float(manual_valence), float(
				manual_arousal), color='blue', s=15)
			self.manual_valence_vec.append(float(manual_valence))
			self.manual_arousal_vec.append(float(manual_arousal))
		self.canvas.draw()

	def clear_plot(self):
		self.createCircle()
		self.manual_arousal_vec.clear()
		self.manual_valence_vec.clear()

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
		self.clicked = False

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
		controlLayout.addWidget(self.playButton)
		controlLayout.addWidget(self.positionSlider)
  
		open_btn_layout = self.open_multimedia_btn
		open_btn_layout.addWidget(openBtn)

		layout = self.multimedia
		layout.addWidget(videoWidget)
		layout.addWidget(self.errorLabel)

		self.mediaPlayer.setVideoOutput(videoWidget)
		self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
		self.mediaPlayer.positionChanged.connect(self.positionChanged)
		self.mediaPlayer.durationChanged.connect(self.durationChanged)
		self.current_vol = 70
		self.mediaPlayer.setVolume(self.current_vol)
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
  
		# set the layout
		va_layout = self.va_plot
		va_layout.addWidget(self.toolbar)
		va_layout.addWidget(self.canvas)
  
		self.info_annotate_button.clicked.connect(self.openInfoBox)
		
		self.figure.canvas.mpl_connect(
			"button_press_event", self.annotateOnClick)

		self.save_CSV_button.clicked.connect(self.saveAsCSV)

		# For re-annotation
		self.reannotate_button.clicked.connect(self.reannotate)

		# Play media from start
		self.playfromStart.clicked.connect(self.playFromStart)
  
		# Forward and Backward buttons
		self.forward.clicked.connect(self.forward3Sec)
		self.backward.clicked.connect(self.backward3Sec)

		# Volume buttons
		self.inc_vol.clicked.connect(self.increaseVolume)
		self.dec_vol.clicked.connect(self.decreaseVolume)

		# Playback speed
		self.playback_slider.setMinimum(-4)
		self.playback_slider.setMaximum(4)
		self.playback_slider.setValue(0)
		self.playback_slider.setTickInterval(1)
		self.playback_slider.setTickPosition(QSlider.TicksBelow)
		self.playback_slider.valueChanged.connect(self.changePlaybackSpeed)

		self.homeB = self.findChild(
			QtWidgets.QPushButton, 'home_button_annotator')
		self.homeB.clicked.connect(self.goto_home)

		# to keep calling the positionChanged function in order to update the time and store it
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.positionChanged)
		self.timer.start(10)

		# to keep calling the autoClick function in order to keep clicking in regular intervals
		self.timerClick = QTimer(self)
		self.timerClick.timeout.connect(lambda: self.autoClicking(None))
		self.timerClick.start(20)
  
	def openInfoBox(self):
		msg = QMessageBox()
		msg.setWindowTitle("Information")
		msg.setText("The annotation section allows users to mark the valence and arousal values of the speech or video signal as a function of time. Deep Learning models are very data-centric and in order to train and be useful at predictions they require a good amount of data to learn from. Over the web or in any public platform the data available for DL training is quite less. There also doesn’t exist any way using which we can create a dataset of emotions according to our needs. The annotation feature of EmotionGUI makes this possible. We can generate CSV files of the emotional data right in the software just by playing and annotating the media right there. This feature will enable the researchers or the open-source world in general, to get good-quality data to train their models.")
		msg.setIcon(QMessageBox.Information)
		msg.setStandardButtons(QMessageBox.Cancel)
		msg.setDetailedText("- Open an mp4 or wav file from the testing/WAV Files or testing/MP4 Files folder by clicking on the 'Open Video/Audio File' button. \n- Play the media and annotate the emotions on the V-A plot on the rightSave the valence-arousal points into a csv file by clicking 'Save as CSV' button. \n- If you select a WAV file, you will be able to see its corresponding spectrogram below the media window for more visual cues")
		msg.exec_()

	def audioVis(self, filepath):
		# a figure instance to plot on
		self.audio_figure = plt.figure()

		# this is the Canvas Widget that displays the `figure`
		# it takes the `figure` instance as a parameter to __init__
		self.audio_canvas = FigureCanvas(self.audio_figure)

		# this is the Navigation widget
		self.audio_toolbar = NavigationToolbar(self.audio_canvas, self)

		axes = self.audio_figure.subplots()
		plt.subplot(211)
		samplingFrequency, signalData = wavfile.read(filepath)
		# Plot the signal read from wav file
		plt.title('Spectrogram', fontsize=9)
		# plt.rcParams["figure.figsize"] = [10, 10]
		plt.plot(signalData)
		plt.xlabel('Sample')
		plt.ylabel('Amplitude')

		plt.subplot(212)
		axes.tick_params(axis='both', which='major', labelsize=8)
		axes.tick_params(axis='both', which='minor', labelsize=8)
		plt.specgram(signalData, Fs=samplingFrequency)
		plt.xlabel('Time')
		plt.ylabel
		plt.subplots_adjust(left=0.130, right=0.950,
							top=0.945, bottom=0.099, hspace=0.265)

		self.audio_layout = self.multimedia
		self.audio_layout.addWidget(self.audio_toolbar)
		self.audio_layout.addWidget(self.audio_canvas)

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

		self.axes.tick_params(axis='both', which='major', labelsize=10)
		self.axes.tick_params(axis='both', which='minor', labelsize=10)

		plt.xlabel("Valence")
		plt.ylabel("Arousal")
		plt.tight_layout()

		# V-A plot basic landmark emotions coordinates
		self.landmarkEmotions = ['angry', 'afraid', 'sad', 'bored', 'excited',
								 'interested', 'happy', 'pleased', 'relaxed', 'content']
		self.landmarkValence = (-0.7, -0.65, -0.8, -0.1, 0.37,
								0.2, 0.5, 0.35, 0.6, 0.5)
		self.landmarkArousal = (0.65, 0.5, -0.15, -0.45, 0.9,
								0.7, 0.5, 0.35, -0.3, -0.45)

		for point in range(len(self.landmarkEmotions)):
			self.axes.scatter(self.landmarkValence,
							  self.landmarkArousal, color='k', s=15)
			self.axes.text(self.landmarkValence[point] + 0.02, self.landmarkArousal[point] + 0.02,
						   self.landmarkEmotions[point], fontsize='large')

		self.axes.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
		self.axes.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)

		self.canvas.draw()

	def annotateOnClick(self, event):
		if self.playButton.isEnabled():
			self.clicked = True
			print(round(event.xdata, 2), round(event.ydata, 2))
			if event is not None and (event.xdata >= -1 and event.xdata <= 1 and event.ydata >= -1 and event.ydata <= 1):
				self.axes.scatter(round(event.xdata, 2), round(
					event.ydata, 2), color='red', s=15)
				self.canvas.draw()
				self.savePoints(round(event.xdata, 2), round(event.ydata, 2))
				self.figure.canvas.mpl_connect(
				"motion_notify_event", self.autoClicking)
			else:
				self.count_out_of_bounds += 1
				self.time_points.append(self.seconds)
				self.valence_points.append(None)
				self.arousal_points.append(None)
				self.out_of_bounds_lbl.setText(
					"You have clicked out of the annotation model {} times. Do you want to re-annotate?".format(self.count_out_of_bounds))
				self.figure.canvas.mpl_connect(
				"motion_notify_event", self.autoClicking)

	def reannotate(self):
		# clear the plot and all the time and emotional coordinates list
		self.clear_plot()
		self.time_points = []
		self.valence_points = []
		self.arousal_points = []
		self.count_out_of_bounds = 0
		self.clicked = False
		# pause the media and begin again
		self.setPosition(0)
		self.mediaPlayer.pause()
		self.out_of_bounds_lbl.setText(
			"You can now begin the annotation again")

	def savePoints(self, xdata, ydata):
		self.valence_points.append(xdata)
		self.arousal_points.append(ydata)
		self.time_points.append(self.seconds)

	def saveAsCSV(self):
		header = ["Time", "Valence", "Arousal"]

		# this opens a dialog box to save the file
		filename = 'annotated-file'
		filename, _ = QFileDialog.getSaveFileName(
			self, "Save CSV file", filename, "CSV Files (*.csv)"
		)
  
		for i in range(len(self.time_points)):
			if i != 0:
					if self.time_points[i] == self.time_points[i-1] or float(self.time_points[i]) < float(self.time_points[i-1]):
						self.time_points[i] = round(float(self.time_points[i-1]) + random.uniform(0.02, 0.07), 2)

		if filename:
			with open(filename, 'w+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(header)
				rows = [(str(time), str(valence), str(arousal)) for time, valence, arousal in zip(self.time_points,
																								  self.valence_points, self.arousal_points)]
				writer.writerows(rows)

	def clear_plot(self):
		self.createCircle()
		self.out_of_bounds_lbl.clear()
		self.valence_points.clear()
		self.arousal_points.clear()

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

	def playFromStart(self):
		self.setPosition(0)
		self.mediaPlayer.play()

	def play(self):
		self.notPlaying = True
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

	def autoClicking(self, event):
		if self.playButton.isEnabled() and self.clicked == True:
			if event is not None and (event.xdata != None and event.ydata != None):
				print(round(event.xdata, 2), round(event.ydata, 2))
				if (event.xdata >= -1 and event.xdata <= 1 and event.ydata >= -1 and event.ydata <= 1):
					time_elapsed = self.seconds
					if float(time_elapsed) <= float(self.mediaSeconds) / 3:
						print(self.mediaSeconds)
						color = 'red'
						opacity = 1 - (float(time_elapsed) / (float(self.mediaSeconds) / 3))
						if opacity < 0:
							opacity = 0.3
						elif opacity > 1:
							opacity = 1
					elif float(time_elapsed) > float(self.mediaSeconds) / 3 and float(time_elapsed) <= 2 * float(self.mediaSeconds) / 3:
						color = 'yellow'
						opacity = 1 - ((float(time_elapsed) - (float(self.mediaSeconds) / 3)) / 3)
						print('yellow', opacity)
						if opacity < 0:
							opacity = 0.3
						elif opacity > 1:
							opacity = 1
					elif float(time_elapsed) > 2 * float(self.mediaSeconds) / 3:
						color = 'blue'
						opacity = 1 - ((float(time_elapsed) - 2 * (float(self.mediaSeconds) / 3)) / 3)
						print('blue', opacity)
						if opacity < 0:
							opacity = 0.3
						elif opacity > 1:
							opacity = 1

					self.axes.scatter(round(event.xdata, 2), round(
						event.ydata, 2), color=color, s=18, alpha=opacity)
					self.canvas.draw()
					self.savePoints(round(event.xdata, 2),
									round(event.ydata, 2))
				else:
					self.count_out_of_bounds += 1
					self.time_points.append(self.seconds)
					self.valence_points.append(None)
					self.arousal_points.append(None)
					self.out_of_bounds_lbl.setText(
						"You have clicked out of the annotation model {} times. Do you want to re-annotate?".format(self.count_out_of_bounds))


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
		# this gives seconds upto two decimal places
		self.seconds = "{:.2f}".format(time_in_seconds)

	def durationChanged(self, duration):
		self.positionSlider.setRange(0, duration)
		mtime = QTime(0, 0, 0, 0)
		mtime = mtime.addMSecs(self.mediaPlayer.duration())
		self.mediaSeconds = mtime.minute() * 60 + mtime.second()
		self.elbl.setText(mtime.toString())

	def setPosition(self, position):
		self.mediaPlayer.setPosition(position)

	def increaseVolume(self):
		self.current_vol += 5
		if self.current_vol > 100:
			self.current_vol = 100
		self.mediaPlayer.setVolume(self.current_vol)

	def decreaseVolume(self):
		self.current_vol -= 5
		if self.current_vol < 0:
			self.current_vol = 0
		self.mediaPlayer.setVolume(self.current_vol)
  
	def forward3Sec(self):
		self.mediaPlayer.setPosition(self.mediaPlayer.position() + 3000)
  
	def backward3Sec(self):
		self.mediaPlayer.setPosition(self.mediaPlayer.position() - 3000)

	def changePlaybackSpeed(self):
		# Get the current value of the slider
		value = self.playback_slider.value()
		# Calculate the new playback rate
		playback_rate = (value / 4) + 1
		# Set the new playback rate on the media player
		self.mediaPlayer.setPlaybackRate(playback_rate)

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
  
		self.info_audio_btn.clicked.connect(self.openInfoBox)

		self.timer.setText('0 sec')
		self.stop_pressed = False
		self.progressBar.setValue(0)

		self.mic = MicrophoneRecorder(self.read_collected)
		self.read_collected.connect(self.update)

		# time (seconds) between reads
		interval = FS/CHUNKSZ
		self.t = QtCore.QTimer()
		self.t.timeout.connect(self.mic.read)
		self.t.start(int(1000/interval))  # QTimer takes ms

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
  
	def openInfoBox(self):
		msg = QMessageBox()
		msg.setWindowTitle("Information")
		msg.setText("The real-time live audio recording feature makes sure that we are able to record audio in real-time for with respect to our need and then use it to visualize the emotional data of the audio using the machine learning model or annotate that recorded audio for very niche needs and use it for further training.")
		msg.setIcon(QMessageBox.Information)
		msg.setStandardButtons(QMessageBox.Cancel)
		msg.setDetailedText("- Click on the 'Live-Audio' button in the main menu to open a new window.\n- Start recording your audio and see its corresponding spectrogram and waveform for visual cues.\n- Stop the audio stream and save it as a WAV file or as a PNG file.\n- Now you can use the generated wav file in the visualize and annotation sections again.")
		self.stop_recording()
		msg.exec_()

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
		# opens a dialog box for saving the file
		self.filename = 'live-audio'
		self.filename, _ = QFileDialog.getSaveFileName(
			self, "Save audio file", self.filename, "WAV Files (*.wav)"
		)
		if self.filename:
			sound_file = wave.open(self.filename, "wb")
			sound_file.setnchannels(1)
			sound_file.setsampwidth(
				self.mic.p.get_sample_size(pyaudio.paInt16))
			sound_file.setframerate(FS)
			sound_file.writeframes(b''.join(self.mic.frames))
			sound_file.close()

	def save_photo(self, audiopath):
		# Read the wav file (mono)
		audiopath = self.filename
		if audiopath:
			samplingFrequency, signalData = wavfile.read(
				audiopath)
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
		widget.setGeometry(QtCore.QRect(150, 125, 1450, 850))
		widget.show()
		sys.exit(app.exec_())
	except:
		print("Exit")
