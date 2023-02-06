import csv
from PyQt5.uic import loadUi
import matplotlib.pyplot as plt
import librosa as lbr
from PyQt5 import QtWidgets
from pydub import AudioSegment, silence
from PyQt5.QtWidgets import QFileDialog, QWidget, QMessageBox
import numpy as np
from main import MainWindow
from utils.file_processing import FileProcessing
import pandas as pd

from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import keras
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Audio

class visualisationScreen(QWidget):
	def __init__(self, widget):
		super(visualisationScreen, self).__init__()
		loadUi('ui/visualize.ui', self)
		self.widget = widget

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
		home = MainWindow(self.widget)
		self.widget.addWidget(home)
		self.widget.setCurrentIndex(self.widget.currentIndex() - 1)
		self.widget.setCurrentWidget(home)