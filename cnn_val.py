'''
cnn_val.py
This file provides an API to apply the CNN model
on the validation set.
'''


import argparse as ap
import os
from utility import getImages, preprocessImages, getText, LETTERS
import pickle


# Get the path of the validation set 
parser = ap.ArgumentParser()
default = (os.path.join(os.getcwd(), "validation_pictures"), os.getcwd() + "\\validation.csv")

parser.add_argument("dir", nargs = '*', default = default)

args = vars(parser.parse_args())
args = list(args["dir"])

validation_image_path = args[0]
validation_label_path = args[1]



def evalCNN(X, Y, classifier):
	'''
	Evaluate the validation data set with the built CNN model

	X: Validation data
	Y: Validation labels
	classifier: The built CNN model
	return: None
	'''
	classifier.evaluate_result(X, Y)



if __name__ == "__main__":

	# Load the training mean, standard deviation and classifier
	data = pickle.load(open( "classifier.pkl", "rb" ))
	train_mean, train_std, classifier = data[0], data[1], data[2]

	# Preprocess the validation data using the training mean and standard deviation
	val_data = preprocessImages(getImages(validation_image_path), train_mean, train_std)
	val_labels = getText(validation_label_path)
	
	evalCNN(val_data, val_labels, classifier)
	




