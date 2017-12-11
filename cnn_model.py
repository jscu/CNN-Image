'''
cnn_model.py
This file creates the prototype of a Convolutional Neural Network
model and contains some of the common functions used by the model.
'''

import tensorflow as tf
import os
from utility import LETTERS
import csv


# (Reference) https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enables the logging information
tf.logging.set_verbosity(tf.logging.INFO)


def getActivationFunction(activation_type):
	'''
	Return the activation function according to the activation type

	activation_type: Name of activation function
	return: The corresponding activation function
	'''
	if activation_type == "relu":
		return tf.nn.relu
	elif activation_type == "relu6":
		return tf.nn.relu6
	elif activation_type == "crelu":
		return tf.nn.crelu
	elif activation_type == "elu":
		return tf.nn.elu
	elif activation_type == "softplus":
		return tf.nn.softplus
	elif activation_type == "softsign":
		return tf.nn.softsign
	elif activation_type == "sigmoid":
		return tf.sigmoid
	elif activation_type == "tanh":
		return tf.tanh
	else:
		print ("Unrecognized activation type. Please try again.\n")
		return None


def getOptimizer(optimizer_type):
	'''
	Return the optimizer according to the optimizer type 
	
	optimizer_type: Name of optimizer
	return: The corresponding optimizer object
	'''
	if optimizer_type == 'adam':
		return tf.train.AdamOptimizer
	elif optimizer_type == 'adagrad':
		return tf.train.AdagradOptimizer
	elif optimizer_type == 'gradient':
		return tf.train.GradientDescentOptimizer
	else:
		print ("Unrecognized optimizer type. Please try again.\n")
		return None


def getLossFunction(loss_function_type):
	'''
	Return the loss function according to the loss type 
	
	loss_function_type: Name of loss function
	return: The corresponding loss function
	'''
	if loss_function_type == 'l1':
		return tf.losses.absolute_difference
	elif loss_function_type == 'l2':
		return tf.squared_difference
	elif loss_function_type == 'hinge':
		return tf.losses.hinge_loss
	elif loss_function_type == 'log':
		return tf.losses.log_loss
	elif loss_function_type == 'sigmoid':
		return tf.losses.sigmoid_cross_entropy
	elif loss_function_type == 'softmax':
		return tf.nn.softmax_cross_entropy_with_logits
	elif loss_function_type == 'softmax_sparse':
		return tf.losses.sparse_softmax_cross_entropy
	else:
		print ("Unrecognized loss function type. Please try again.\n")
		return None


def getRegularizationFunction(reg_function_type):
	'''
	Return the regularization function according to the regularization type 
	
	reg_function_type: Name of regularization function
	return: The corresponding regularization function or None
	'''
	if reg_function_type == 'l2':
		return tf.contrib.layers.l2_regularizer(scale = 0.5)
	else:
		return None


def getLoss(loss_type, word_logits, word_labels):
	'''
	The loss that CNN tries to minimize 
	
	loss_type: Loss function
	word_logits: The prediction of the CNN model
	word_labels: The actual target matrix
	return: The combined loss for all the ten characters
	'''
	char1_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 0], logits = word_logits[:, 0, :]))
	char2_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 1], logits = word_logits[:, 1, :]))
	char3_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 2], logits = word_logits[:, 2, :]))
	char4_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 3], logits = word_logits[:, 3, :]))
	char5_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 4], logits = word_logits[:, 4, :]))
	char6_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 5], logits = word_logits[:, 5, :]))
	char7_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 6], logits = word_logits[:, 6, :]))
	char8_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 7], logits = word_logits[:, 7, :]))
	char9_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 8], logits = word_logits[:, 8, :]))
	char10_cross_entropy = tf.reduce_mean(loss_type(labels = word_labels[:, 9], logits = word_logits[:, 9, :]))

	loss = char1_cross_entropy + char2_cross_entropy + char3_cross_entropy + char4_cross_entropy + char5_cross_entropy + char6_cross_entropy + char7_cross_entropy + char8_cross_entropy + char9_cross_entropy + char10_cross_entropy
	return loss


class CNN:
	'''
	The Convolutional Neural Networks class
	'''

	def __init__(self, input_layer, output,layers, activation = "relu", loss_type = "softmax", optimizer = "adam", regularization = None,
		batch_size = 64, padding = 'SAME', learning_rate = 1e-4, iteration = 10000, batch_norm = False, drop_out = False, drop_out_rate = 0):

		self.input_layer = input_layer					# The input data
		self.output = output							# The output data
		self.activation = activation 					# The activation function
		self.loss_type = loss_type 						# The loss function
		self.optimizer = optimizer 						# The optimizer
		self.regularization = regularization 			# The regularization function
		self.batch_size = batch_size 					# The number of batches used for training
		self.padding = padding 							# The padding size added to the filter
		self.learning_rate = learning_rate 				# The learning rate for training
		self.iteration = iteration 						# The number of iterations for training
		self.batch_norm = batch_norm 					# Whether to apply batch normalization to a layer
		self.drop_out = drop_out 						# Whether to apply dropout to a layer
		self.drop_out_rate = drop_out_rate  			# The drop out rate
		self.stack_layers = [] 							# The list that stacks all the output neurons

		self.classifier = tf.estimator.Estimator(model_fn = self.buildModel, model_dir = os.path.join(os.getcwd(), "model"))	#The estimator object used for training, evaluation and prediction

		self.layers = layers 							# The network architecture with the form of [("C", a, b), ("P", [c,d], [e,f]), ("R"),("FC", g),("O", h)],
		                                            	# where "C" denotes convolutional layer, "P" denotes pooling layer, "R" denotes reshape layer, "FC" denotes
		                                            	# fully connected layer, "O" denotes output layer. For a,b, they represent the filter size and kernel size in convolutional layers.
		                                            	# For [c,d] and [e,f], they represent the pool size and strides in pooling layers. For g, it represents number of neurons in
		                                            	# the FC layers. Lastly for h, it represents the number of output neurons. There can be more than one layer for every layer types 
		                                            	# except the reshape layer.
		                                            	 



	def createLayer(self, curr_layer, prev_layer):
		'''
		Creates the layers according to the layer type to form the CNN model

		curr_layer: Current layer that has specification as mentioned above
		prev_layer: Previously created layer
		return: The CNN layer
		'''

		# The Convolutional layer
		if curr_layer[0] == "C":
			temp_layer = None
			
			# If the layer specification has a length of three, no strides parameters are supplied
			if len(curr_layer) == 3:
				temp_layer = tf.layers.conv2d(inputs = prev_layer, filters = curr_layer[1], kernel_size = curr_layer[2], padding = self.padding,
											  kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01), 
											  kernel_regularizer = getRegularizationFunction(self.regularization))

			if len(curr_layer) == 4:
				temp_layer = tf.layers.conv2d(inputs = prev_layer, filters = curr_layer[1], kernel_size = curr_layer[2], 
											  strides = curr_layer[3], padding = self.padding, 
											  kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
					                          kernel_regularizer = getRegularizationFunction(self.regularization))

			# Determine whether to apply batch normalization
			if self.batch_norm is True:
				temp_layer = tf.layers.batch_normalization(temp_layer)

			act_layer = getActivationFunction(self.activation)(temp_layer)

			return act_layer

		# The Pooling layer
		elif curr_layer[0] == "P":
			prev_layer = tf.layers.max_pooling2d(inputs = prev_layer, pool_size = curr_layer[1], strides = curr_layer[2], padding = self.padding)

			# Determine whether to apply dropout
			if self.drop_out is True:
				prev_layer = tf.layers.dropout(prev_layer, rate = self.drop_out_rate)

			return prev_layer

		# The reshape layer
		elif curr_layer[0] == "R":
			prev_shape = prev_layer.get_shape()

			return tf.reshape(prev_layer, [-1, int(prev_shape[1]) * int(prev_shape[2]) * int(prev_shape[3])])

		# The fully connected layer
		elif curr_layer[0] == "FC":
	
			return tf.layers.dense(inputs = prev_layer, units = curr_layer[1], activation = getActivationFunction(self.activation),
				kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01), 
				kernel_regularizer = getRegularizationFunction(self.regularization))

		# The output layer
		elif curr_layer[0] == "O":

			self.stack_layers.append(tf.layers.dense(inputs = prev_layer, units = curr_layer[1]))
			return tf.layers.dense(inputs = prev_layer, units = curr_layer[1])

		else:
			print ("Unrecognized layer type. Please try again.\n")
			return None


	def buildModel(self, features, labels, mode):
		'''
		The function is responsible for applying the CNN layers for training, evaluation and prediction

		features: The input data
		labels: The input labels
		mode: The mode of the classifier. Either training, evluation or predication.
		return: The results of the CNN model
		'''	
		self.stack_layers = []
		prev_layer = features["x"]
		
		# Keep creating layers on top of each other
		for curr_layer in self.layers:
			if curr_layer[0] == "O":
				self.createLayer(curr_layer, prev_layer)
			else:
				prev_layer = self.createLayer(curr_layer, prev_layer)

		# Stack all the output neurons
		prev_layer = tf.stack(self.stack_layers, axis = 1)


		predictions = {
			# The character class predict by the model
			"classes": tf.argmax(input = prev_layer, axis = 2)
		}

		#Predict the character class for predict mode
		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)


		img_aucc = tf.metrics.accuracy(labels = tf.argmax(labels, axis = 2), predictions = tf.argmax(input = prev_layer, axis = 2))


		img_recall = tf.metrics.recall(labels = tf.argmax(labels, axis = 2), predictions = tf.argmax(input = prev_layer, axis = 2))


		img_precision = tf.metrics.precision(labels = tf.argmax(labels, axis = 2), predictions=tf.argmax(input = prev_layer, axis = 2))


		# Three different metrics used by evaluation mode
		eval_metric_ops = {

			"image accuracy": img_aucc,

			"recall": img_recall,

			"precision": img_precision
		}
		
		# Scalars to display in Tensorboard
		with tf.name_scope('accuracy_of_labels'):
			correct_prediction = tf.equal(tf.round(tf.nn.softmax(prev_layer)), tf.round(labels))
			accuracy_lab = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		tf.summary.scalar('accuracy_of_labels', accuracy_lab)

		with tf.name_scope('accuracy_of_images'):
			all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
			accuracy_img = tf.reduce_mean(all_labels_true)

		tf.summary.scalar('accuracy_of_images', accuracy_img)

		with tf.name_scope('precision'):
			_, pre = img_precision

		tf.summary.scalar('precision', pre)

		with tf.name_scope('recall'):
			_, rec = img_recall

		tf.summary.scalar('recall', rec)

		# The loss value minimized by the optimizer
		loss = getLoss(getLossFunction(self.loss_type), prev_layer, labels)
		
		# Select the optimizer and training the model by minizing the loss value in training mode
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = getOptimizer(self.optimizer)(learning_rate = self.learning_rate)

			train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())

			return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

		# Return the metrics results for evaluation mode
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)


	def train(self):
		'''
		Train the CNN model invoked by other files

		input: None
		return: None
		'''

		# Log the loss
		logging_hook = tf.train.LoggingTensorHook(tensors = {}, every_n_iter = 100)

		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x = {"x": self.input_layer},
			y = self.output,
			batch_size = self.batch_size,
			num_epochs = None,
			shuffle = True)

		# Train the model with the above specification supplied
		self.classifier.train(input_fn = train_input_fn, steps = self.iteration, hooks = [logging_hook])
		



	def evaluate_result(self, eval_data, eval_labels):
		'''
		Evaluate the result based on the evaluation data and write the results to CSV file

		eval_data: Evaluation data
		eval_labels: Evaluation labels
		return: None
		'''

		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x = {"x": eval_data},
			y = eval_labels,
			num_epochs = 1,
			shuffle = False)

		# Evaluate the model with the evaluation data
		eval_results = self.classifier.evaluate(input_fn = eval_input_fn)
		
		csv_file = open(os.getcwd() + "\\validation_results.csv", 'w', newline = '')

		writer = csv.writer(csv_file)

		for k,v in eval_results.items():
			print (str(k) + ": " + str(v))
			print("\n")
			writer.writerow((str(k), str(v)))

		csv_file.close()
		




	def predict_result(self, data):
		'''
		Predict the english words based on the image

		data: Images to predict
		return: None
		'''

		predict_input_fn = tf.estimator.inputs.numpy_input_fn(
			x = {"x": data},
			num_epochs = 1,
			shuffle = False)

		predict_results = list(self.classifier.predict(input_fn = predict_input_fn))

		csv_file = open(os.getcwd() + "\\predict_results.csv", 'w', newline = '')

		writer = csv.writer(csv_file)

		# Write the predicted english words to CSV file
		for i in predict_results:
			text = []
			for j in i["classes"]:
				print (LETTERS[j], end = '')
				text.append(LETTERS[j])
			print("\n")

			writer.writerow(["".join(text)])
			







