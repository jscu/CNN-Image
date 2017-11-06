'''
cnn_classify_image.py
This file provides an API to predict the english 
words from images with the built CNN model.
'''

import argparse as ap
import os
from utility import getImages, preprocessImages
import pickle

# Get the path of the test set
parser = ap.ArgumentParser()
parser.add_argument("dir", nargs = '?', default = (os.path.join(os.getcwd(), "test_pictures")))

args = vars(parser.parse_args())

test_image_path = args["dir"]

def predictCNN(X, classifier):
	'''
	Predict the english words from image

	X: The test image data
	classifier: The built CNN model
	return: None
	'''
	classifier.predict_result(X)



if __name__ == "__main__":

	# Load the training mean, standard deviation and classifier
	data = pickle.load(open( "classifier.pkl", "rb" ))
	train_mean, train_std, classifier = data[0], data[1], data[2]

	# Preprocess the test data using the training mean and standard deviation
	test_data = preprocessImages(getImages(test_image_path), train_mean, train_std)

	predictCNN(test_data, classifier)





	


	