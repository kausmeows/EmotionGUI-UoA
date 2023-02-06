import wave
from PyQt5.uic import loadUi
import pyaudio
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QWidget, QMessageBox
import numpy as np
from main import MainWindow
from utils.spectogram import MicrophoneRecorder
from scipy.io import wavfile

from PyQt5.QtWidgets import (QFileDialog, QWidget)
import pyqtgraph as pg
from utils import audio_processing as AP

FS = 44100  # Hz
CHUNKSZ = 1024  # samples
form_class = uic.loadUiType("ui/audio.ui")[0]
counter = 0

class liveAudioScreen(QMainWindow, QWidget):
	read_collected = QtCore.pyqtSignal(np.ndarray)
	def __init__(self, widget):
		super(liveAudioScreen, self).__init__()
		loadUi('ui/audio.ui', self)	
		self.widget = widget
		self.start_pressed = False
  
		self.start = self.findChild(
		QtWidgets.QPushButton, 'start')
		self.start.clicked.connect(self.start_recording) 
  
		self.info_audio_btn.clicked.connect(self.openInfoBox)

		self.timer.setText('0 sec')
		self.stop_pressed = False
		self.progressBar.setValue(0)

		self.mic = MicrophoneRecorder(self.read_collected)
		self.read_collected.connect(self.update)

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
  
		self.main_menu.clicked.connect(self.goto_home)

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
  
	def start_recording(self):
		self.start_pressed = True
		# time (seconds) between reads
		interval = FS/CHUNKSZ
		self.t = QtCore.QTimer()
		self.t.timeout.connect(self.mic.read)
		self.t.start(int(1000/interval))  # QTimer takes ms

		self.timerrr = QtCore.QTimer()
		self.timerrr.timeout.connect(self.update_counter)
		self.timerrr.start(1000)  # QTimer takes ms

	def stop_recording(self):
		if self.start_pressed != True:
			return
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

	def goto_home(self):
		print("homeeee")
		home = MainWindow(self.widget)
		self.widget.addWidget(home)
		self.widget.setCurrentIndex(self.widget.currentIndex() - 1)
		self.widget.setCurrentWidget(home)