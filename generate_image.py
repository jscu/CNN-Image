'''
generate_image.py
This file gets english words form a list of webpages and
generate the images of the english words and save the 
corresponding labels to a CSV file.
'''

import os, sys
from os import listdir
from os.path import isfile, join
import requests
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from random import randint, uniform
import numpy as np
from utility import LETTERS
import csv
import argparse as ap

# (Reference) https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
# Get the path of the training images and csv from user input
parser = ap.ArgumentParser()
default = ("10", os.path.join(os.getcwd(), "training_pictures\\"), os.getcwd() + "\\training.csv") 
parser.add_argument("dir", nargs='*', default = default)	

args = vars(parser.parse_args())
args = list(args["dir"])

no_of_images = int(args[0])

# Determine to use the user applied or default path based on argument length 
if len(args) == 1:
	training_image_path = default[1]
else:
	training_image_path = args[1]

if len(args) == 1 or len(args) == 2:
	training_label_path = default[2]
else:
	training_label_path = args[2]

# List of text to get the english words
webpages = ["https://www.gutenberg.org/files/1342/1342-0.txt", "http://www.gutenberg.org/cache/epub/3691/pg3691.txt", "https://www.gutenberg.org/files/11/11-0.txt",
				"http://www.gutenberg.org/cache/epub/2542/pg2542.txt", "http://www.gutenberg.org/cache/epub/84/pg84.txt",
				"http://www.gutenberg.org/cache/epub/844/pg844.txt", "http://www.gutenberg.org/cache/epub/345/pg345.txt",
				"http://www.gutenberg.org/cache/epub/5200/pg5200.txt", "https://www.gutenberg.org/files/76/76-0.txt", "http://www.gutenberg.org/cache/epub/1952/pg1952.txt",
				"https://www.gutenberg.org/files/98/98-0.txt", "https://www.gutenberg.org/files/74/74-0.txt", "https://www.gutenberg.org/files/2591/2591-0.txt",
				"http://www.gutenberg.org/cache/epub/3207/pg3207.txt", "http://www.gutenberg.org/cache/epub/46/pg46.txt"]


def createDictionary():
	'''
	Create a dictionary of strings consist of only four to ten letters from the webpages

	return:	Matrix of english words.
	'''
	list_of_words = []
	for web in webpages:
		page = requests.get(web)
		words = page.content.split()
		for word in words:
			temp_word = str(word, "utf-8")

			# Determine if the word only consist of letters and has length of between four to ten and not in the dictionary
			if temp_word.isalpha() and temp_word not in list_of_words and len(temp_word) <= 10 and len(temp_word) >= 4:
				list_of_words.append(temp_word)

	return list_of_words


def getRandomNumbers(lower_limit, upper_limit, no_of_numbers = 1):
	'''
	Get random number/numbers within a given lower and upper limit

	lower_limit: Lower limit of random number
	upper_limit: Upper limit of random number
	no_of_numbers: Number of random numbers to generate
	return: A random number or a tuple of random numbers
	'''
	if no_of_numbers == 1:
		return randint(lower_limit, upper_limit)
	else:
		return tuple(randint(lower_limit, upper_limit) for _ in range(no_of_numbers))


def getWords(dictionary):
	'''
	Get a string randomly from the dictionary or generate a string with random letters 
	
	dictionary: A dictionary of english words
	return: A random word from the dictionary or a word of random letters
	'''
	if dictionary != LETTERS:
		return dictionary[getRandomNumbers(0, len(dictionary) - 1)]
	else:
		# (Reference) https://stackoverflow.com/questions/16060899/alphabet-range-python
		# Generate a word with four to ten random letters
		no_of_letters = getRandomNumbers(4, 10)
		return ''.join(np.random.choice(dictionary, no_of_letters))



def generateImages(img_save_path, letter_save_path, font_type, number_of_images = 1, size = (128, 64), random_words = False):
	'''
	Generate images with english word and save it to a path
	
	img_save_path: Save path of image
	letter_save_path: Save path of letters
	font_type: Type of font for the words
	number_of_images: Number of images to generate
	size: Size of images
	random_words: Whether to generate word with random letters or not
	return: Images with english word or word with random letters with random background and text color
	'''
	dictionary = []

	if not os.path.exists(img_save_path):
		os.makedirs(img_save_path)

	if random_words is False:
		dictionary = createDictionary()

		# This is for reloading the already saved dictionary to save time
		#import pickle
		#dictionary = pickle.load(open('words.pkl', "rb" ))
	else:
		dictionary = LETTERS

	csv_file = open(letter_save_path, 'w', newline='')

	writer = csv.writer(csv_file)

	for i in list(range(number_of_images)):

		# (Reference) https://stackoverflow.com/questions/16373425/add-text-on-image-using-pil 
		# Choose a random background color
		img = Image.new('RGB', size, getRandomNumbers(126, 255, 3))

		draw = ImageDraw.Draw(img)

		text_color = getRandomNumbers(0, 125, 4)
	
		text = getWords(dictionary)

		# Remove the selected text from dictionary
		if text in dictionary:
			dictionary.remove(text)

		text = text.encode("utf8").decode("cp950", "ignore")
		
		# If the length of word is ten and in uppercase, shrink the font size
		if len(text) == 10 and text.isupper():
			font = ImageFont.truetype(font_type, size[0] // 10 + 6)
		else:
			font = ImageFont.truetype(font_type, size[0] // 10 + 8)

		# (Reference) https://nicholastsmith.wordpress.com/2017/10/14/deep-learning-ocr-using-tensorflow-and-python/
		# Calculate the offset
		location = (((size[0] - font.getsize(text)[0]) // 2), ((size[1] - font.getsize(text)[1]) // 2))

		draw.text(location, text, text_color, font = font)

		img.save(img_save_path + str(i) + '.png')

		print ("Image#" + str(i + 1) + " was created and saved")

		# Write the text and its length to CSV file
		writer.writerow([text])

	csv_file.close()


if __name__ == "__main__":
	generateImages(training_image_path, training_label_path, "arial.ttf", no_of_images)
	


	