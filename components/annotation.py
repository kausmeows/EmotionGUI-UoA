import os
import random
import csv
from PyQt5.uic import loadUi
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDir, Qt, QUrl, QTime
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QPushButton, QMessageBox
from PyQt5.QtCore import QDir, Qt, QUrl, QTimer
import numpy as np
from main import MainWindow
from scipy.io import wavfile

from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QFileDialog, QLabel,
							 QPushButton, QSizePolicy, QSlider, QStyle)
from PyQt5.QtWidgets import QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class annotationScreen(QMainWindow):
	def __init__(self, widget):
		super(annotationScreen, self).__init__()
		loadUi('ui/annotate.ui', self)
		self.widget = widget

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
	
		_, fileExtension = os.path.splitext(fileName)
		fileExtension = fileExtension[1:] # remove the dot

		if fileName != '':
			self.mediaPlayer.setMedia(
				QMediaContent(QUrl.fromLocalFile(fileName)))
			self.playButton.setEnabled(True)
			if(fileExtension == 'wav'):
				self.audioVis(fileName)

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
		home = MainWindow(self.widget)
		self.widget.addWidget(home)
		self.widget.setCurrentIndex(self.widget.currentIndex() - 1)
		self.widget.setCurrentWidget(home)