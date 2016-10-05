import numpy as np 
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation 
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import random


'''
Data obtained from flappy.py
playery = -1
playerVelY = -1
pipeHeight = -1
pipeX = -1
train data works in above order
'''
#Initialize the classifier
clf = GaussianNB()

single_train_data = [-1,-1,-1,-1]

#to store when the last click occured 
lastclick = 0

features_train = []
labels_train = []

def generateReply(playery, playerVelY, pipeHeight, pipeX):
	#To return the new data after processing
	
	global single_train_data

	single_train_data = [playery, playerVelY, pipeHeight, pipeX]

	try:
		result = clf.predict()
		return result

	except:

		if random.random() <0.5:
			return 1
		else:
			return 0


def generateData(result):
	#What to do after we get data

	global features_train, labels_train

	result = ~(result)

	features_train.append(single_train_data)
	labels_train.append(result)


def fit_mldata():

	global features_train, labels_train

	features_train = np.array(features_train)
	labels_train = np.array(labels_train)

	clf.fit(features_train, labels_train)

	joblib.dump(clf, 'data_generated.pkl')

	print "Data fitted"