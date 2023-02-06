import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QPixmap

from PyQt5.QtWidgets import (QApplication, QPushButton)
from PyQt5.QtWidgets import QMainWindow

class MainWindow(QMainWindow, QPushButton):
	def __init__(self, widget):
		super(MainWindow, self).__init__()
		self.widget = widget
  
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
		from components.visualize import visualisationScreen
		visualise = visualisationScreen(self.widget)
		self.widget.addWidget(visualise)
		self.widget.setCurrentIndex(self.widget.currentIndex() + 1)

	def goto_annotate(self):
		from components.annotation import annotationScreen
		annotate = annotationScreen(self.widget)
		self.widget.addWidget(annotate)
		self.widget.setCurrentIndex(self.widget.currentIndex() + 1)

	def goto_liveAudio(self):
		from components.live_audio import liveAudioScreen
		liveAudio = liveAudioScreen(self.widget)
		self.widget.addWidget(liveAudio)
		self.widget.setCurrentIndex(self.widget.currentIndex() + 1)


if __name__ == "__main__":
	try:
		app = QApplication(sys.argv)
		widget = QtWidgets.QStackedWidget()

		home = MainWindow(widget)
		widget.addWidget(home)
		widget.setGeometry(QtCore.QRect(150, 125, 1450, 850))
		widget.show()
		sys.exit(app.exec_())
	except:
		print("Exit")
