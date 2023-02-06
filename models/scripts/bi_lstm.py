import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import librosa as lbr
from pydub import AudioSegment, silence
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Audio

class BiLSTM():
	def __init__(self, model_path):
		self.model_path = model_path
		Audio(self.model_path)

	# Preprocessing & Feature Extraction
	def silenceStampExtract(self, audioPath, length):
		myaudio = AudioSegment.from_wav(audioPath)
		slc = silence.detect_silence(
			myaudio, min_silence_len=1000, silence_thresh=-32)
		slc = [((start/1000), (stop/1000))
			for start, stop in slc]  # convert to sec
		slc = np.array([item for sublist in slc for item in sublist])  # flatten
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

		##################Energy##################
		rms = ((lbr.feature.rms(x, frame_length=frameLength,
			hop_length=frameLength, center=True))[0])
		rms = 20*np.log10(rms)

		##################F0##################
		f0Result = lbr.yin(x, 50, 300, sr, frame_length=frameLength*4)

		##################MFCC##################
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
		self.currentDf = self.featureExtract(self.model_path)
		self.tagDf = self.silenceStampExtract(self.model_path, self.currentDf.shape[0])
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
			# inv transform the predicted value
		# 	yPredict = scaler.inverse_transform(inv_yPredict.reshape(-1, 1))
		# 	yPredict = yPredict[:, 0]
		# else:
			yPredict = arousalModel.predict(test_X)

		a_pred_test_list = [i for i in yPredict]


		# make a prediction
		if transformTarget:
			inv_yPredict = valenceModel.predict(test_X)
			# inv transform the predicted value
		# 	yPredict = scaler.inverse_transform(inv_yPredict.reshape(-1, 1))
		# 	yPredict = yPredict[:, 0]
		# else:
			yPredict = valenceModel.predict(test_X)

		v_pred_test_list = [i for i in yPredict]


		time_array = self.currentDf[['Time']].to_numpy()
		time_array = time_array[24:len(time_array)]

		csvOutput = np.column_stack((time_array, v_pred_test_list, a_pred_test_list))

		# csvFilename = '/Users/kaustubh/Desktop/EmotionGUI-UoA/output_files/csv/predicted/predicted_emotions.csv'

		# # newPath = os.path.join(
		# # 	os.getcwd(), r"output_files/csv/predicted")
		# # os.chdir(newPath)

		# with open(csvFilename, 'w', newline='', encoding='utf-8') as file:
		# 	writer = csv.writer(file)
		# 	writer.writerow(["Time", "Valence", "Arousal"])

		# 	for coordinate in csvOutput.round(decimals=2):
		# 		writer.writerow(coordinate)


# findValues = BiLSTM(
# 	'/Users/kaustubh/Desktop/EmotionGUI-UoA/testing/WAV_Files/JLCO_male2_confident_13a_1.wav')
# findValues.dataframes()
# findValues.predict()