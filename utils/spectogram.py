"""
Tested on Linux with python 3.7
Must have portaudio installed (e.g. dnf install portaudio-devel)
pip install pyqtgraph pyaudio PyQt5
"""
from utils import audio_processing as AP
import sys
import atexit
from PyQt5 import QtCore, uic, QtWidgets
import pyaudio
import numpy as np
import threading
import wave
from PyQt5.uic import loadUi
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import matplotlib
import matplotlib.pyplot as plot
matplotlib.use('Qt5Agg')

from scipy.io import wavfile
matplotlib.use('qt5agg')
form_class = uic.loadUiType("ui/audio.ui")[0]


FS = 44100  # Hz
CHUNKSZ = 1024  # samples
counter = 0


class MicrophoneRecorder():
    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=FS,
                                  input=True,
                                  frames_per_buffer=CHUNKSZ,
                                  #stream_callback=self.new_frame
                                  )
        self.stream.start_stream()
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def read(self):
        data = self.stream.read(CHUNKSZ, exception_on_overflow=False)
        self.frames.append(data)
        y = np.fromstring(data, 'int16')
        self.signal.emit(y)
    
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class SpectrogramWidget(QtWidgets.QMainWindow, form_class, MicrophoneRecorder):
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(SpectrogramWidget, self).__init__()
        loadUi('ui/audio.ui', self)

        self.timer.setText('0 sec')
        self.stop_pressed = False
        self.progressBar.setValue(0)

        self.mic = MicrophoneRecorder(self.read_collected)

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

        # makes sure you can save by adding up all the chunks to make the complete frame
        self.mic.frames.append(chunk)

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
        plot.subplot(211)
        plot.title('Spectrogram')
        plot.plot(signalData)
        plot.xlabel('Sample')
        plot.ylabel('Amplitude')
        plot.subplot(212)
        plot.specgram(signalData, Fs=samplingFrequency)
        plot.xlabel('Time')
        plot.ylabel('Frequency')
        plot.show()
    
    def update_counter(self):
        global counter
        if self.stop_pressed != True:
            counter += 1
            self.timer.clear()
            self.progressBar.setValue(counter)
            self.timer.setText(str(counter) + ' sec')

        else:
            return


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    w = SpectrogramWidget()
    w.read_collected.connect(w.update)

    mic = MicrophoneRecorder(w.read_collected)

    # time (seconds) between reads
    interval = FS/CHUNKSZ
    t = QtCore.QTimer()
    t.timeout.connect(mic.read)
    t.start(1000/interval)  # QTimer takes ms

    app.exec_()
    # mic.close()
