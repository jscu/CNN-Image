'''
utility.py
This file provides some common functions used by 
other files. These functions facilitate the preprocessing
and retrieval stages of the input images.
'''

import numpy as np
import os
import cv2
import string

# Define the 52 English lowercase and uppercase characters and an additional space character
LETTERS = ['\x00'] + list(string.ascii_letters)	

def getImages(path):
	'''
	Helper function for getting images from a path

	path: Path to the images
	return:	An array of images
	'''
	images = []
	training_names = os.listdir(path)

	# Read the images in grayscale mode
	for i in training_names:
		images.append(cv2.imread(os.path.join(path, i), cv2.IMREAD_GRAYSCALE))	
	images = np.stack(images)

	return images

def preprocessImages(data, mean, std):
	'''
	Helper function for processing the images

	data: List of images
	mean: Mean of the standardization process
	std: Standard Deviation of the standardization process
	return:	Standardized images in the correct shape
	'''
	data = data.astype('float32')
	data -= mean
	data /= std

	# Add one more dimension to the images depending on number of color channels 
	if len(data.shape) == 3:	
		return data.reshape(-1, data.shape[1], data.shape[2], 1)
	else:
		return data.reshape(-1, data.shape[1], data.shape[2], 3)



def getText(path, word_length = 10):
	'''
	Helper function for processing the words into matrix

	path: Path to the file which stores the english words
	word_length: Length of the longest word. Default is 10
	return:	3D Matrix with 0's and 1's representing the english words
	'''
	words = []

	with open(path) as f:
		for word in f:
			word = word.strip()

			# Pad the word with spaces until the length reaches 10
			word = (word + ('\x00' * (word_length - len(word))))

			# Transform to a one-hot 2D matrix (length of word, 53 characters)				
			one_hot = [[0 if i != j else 1 for i in LETTERS] for j in word]	

			words.append(one_hot)

	words = np.stack(words)
	words = words.astype(np.float32)

	return words