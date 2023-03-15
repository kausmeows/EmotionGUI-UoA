import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QPixmap

from PyQt5.QtWidgets import (QApplication, QPushButton)
from PyQt5.QtWidgets import QMainWindow

# Define a class that inherits from QMainWindow and QPushButton
class MainWindow(QMainWindow, QPushButton):
    # Constructor to initialize the class
	def __init__(self, widget):
		super(MainWindow, self).__init__()
		self.widget = widget # Initialize the widget
  
		# Load the user interface from a .ui file
		loadUi('ui/home.ui', self)
		# Load an image file and set it as the pixmap for an image label widget
		pic = QPixmap('static/uoa_logo.png')
		self.imglabel.setPixmap(pic)

		# Find and initialize three push buttons, and connect them to their respective methods
		self.visualiseB = self.findChild(
			QtWidgets.QPushButton, 'visualise_button')
		self.visualiseB.clicked.connect(self.goto_visualize)

		self.annotateB = self.findChild(
			QtWidgets.QPushButton, 'annotate_button')
		self.annotateB.clicked.connect(self.goto_annotate)

		self.liveAudioB = self.findChild(
			QtWidgets.QPushButton, 'liveAudio_button')
		self.liveAudioB.clicked.connect(self.goto_liveAudio)

	# Method that is called when the "visualise" button is clicked
	def goto_visualize(self):
		"""
		It creates an instance of the visualisationScreen class, adds it to the widget stack, and then sets
		the current widget to the visualisationScreen
		"""
		from components.visualize import visualisationScreen
		visualise = visualisationScreen(self.widget)
		self.widget.addWidget(visualise)
		self.widget.setCurrentIndex(self.widget.currentIndex() + 1)

	# Method that is called when the "annotate" button is clicked
	def goto_annotate(self):
		"""
		It creates an instance of the annotationScreen class, adds it to the widget stack, and then sets the
		current widget to the annotationScreen
		"""
		from components.annotation import annotationScreen
		annotate = annotationScreen(self.widget)
		self.widget.addWidget(annotate)
		self.widget.setCurrentIndex(self.widget.currentIndex() + 1)

	# Method that is called when the "live audio" button is clicked
	def goto_liveAudio(self):
		"""
		It creates a new instance of the liveAudioScreen class, adds it to the widget stack, and then sets
		the current widget to the new instance
		"""
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
