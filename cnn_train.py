'''
cnn_train.py
This file provides an API to create a CNN model
and train the model by passing in the images and
the corresponding labels.
'''

import tensorflow as tf
from cnn_model import CNN
import string
import argparse as ap
import os
import numpy as np
from utility import getImages, preprocessImages, getText, LETTERS
import pickle


# Get the path of the training set
parser = ap.ArgumentParser()
default = (os.path.join(os.getcwd(), "training_pictures"), os.getcwd() + "\\training.csv")

# (Reference) https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
parser.add_argument("dir", nargs = '*', default = default)

args = vars(parser.parse_args())
args = list(args["dir"])

# Get the path to training images and labels from the arguments
training_image_path = args[0]
training_label_path = args[1]


def buildCNN(X, Y, layers, activation = "relu", loss_type = "softmax", optimizer = "adam", regularization = None ,
			 batch_size = 64, padding = 'SAME', learning_rate = 1e-3, iteration = 10000, batch_norm = False, 
			 drop_out = False, drop_out_rate = 0):
	'''
	Create a CNN object given the model specifications

	input: Details can be found in CNN class of cnn_model.py
	return: CNN model
	'''

	model = CNN(X, Y ,layers, activation, loss_type, optimizer, regularization, batch_size, padding, 
				learning_rate, iteration, batch_norm, drop_out, drop_out_rate)

	return model


def trainCNN(classifier):
	'''
	Training the classifier given the CNN model

	classifier: CNN model
	return: None
	'''
	classifier.train()


if __name__ == "__main__":

	# Get the training images from path and calculate the mean and standard deviation
	images = getImages(training_image_path)

	train_mean = np.mean(images)
	train_std = np.std(images)

	# The Convolutional Neural Network Architecture that I use. It can be changed according to your own preference.
	layers = [("C", 48, 5), ("P", [2,2], [2,2]),("C", 64, 5), ("P", [2, 1], [2,1]), ("C", 128, 5), 
			  ("P", [2,2], [2,2]), ("R"),("FC", 2048), ("O", 53), ("O", 53), ("O", 53), 
			  ("O", 53),("O", 53),("O", 53),("O", 53),("O", 53),("O", 53), ("O", 53)]


	# Preprocess the training Images and get the training labels and text length
	training_data = preprocessImages(images, train_mean, train_std)
	training_labels = getText(training_label_path)


	classifier = buildCNN(training_data, training_labels, layers, iteration = 100, batch_norm = True, drop_out = True, drop_out_rate = 0.2,
						  learning_rate = 1e-3, batch_size = 64)
	
	# Save the training mean, standard deviation and classifier for evaluation and prediction
	pickle.dump([train_mean, train_std, classifier], open("classifier.pkl", "wb" ))

	trainCNN(classifier)

	
	






