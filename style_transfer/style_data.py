from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.utils import shuffle

import numpy as np
import tensorflow as tf


import glob
import cv2
import os


import tensorflow as tf

#local imports
from data import *
import utils as utils
from CNN import new_fc_layer
from CNN import flatten_layer
from CNN import new_conv_layer
from CNN import initiate


# disables cpu instruction warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#################################
####	global variabless 	#####
#################################

true = True
false = False
none = None


total_iterations = None     #explained later
x = None                    #explained later
y_true = None               #explained later
session = None              #explained later
optimizer = None            #explained later
accuracy = None             #explained later

image_size = 100 									# size of the images
image_size_flat = image_size * image_size  			# Images are stored in one-dimensional arrays of this length.
image_shape = (image_size, image_size)  				# Tuple with height and width of images used to reshape arrays.
num_channels = 1  								# Number of color channels for the images: 1 channel for gray-scale.
num_classes = 2  								# Number of classes, one class for each of 10 digits.
plt_show = true  								# To show the plotted values set to true, to never plot anything set to false
class_zero = "rand"
class_one = "kw"
file_name_identifier = "kw"  					# something distinguishable to tell the two images apart
data_directory = "resized/load/chosen"			# directory to load the train images

#####################
####	Layers	#####
#####################

# Convolutional Layer 1.
filter_size1 = 10  # Convolution filters are filter_size x filter_size pixels. might change this to 0.178 * img_size xxx
num_filters1 = 16  # There are 16 of these filters.
# Convolutional Layer 2.
filter_size2 = 10  # Convolution filters are 5 x 5 pixels.
num_filters2 = 36  # There are 36 of these filters.
# Fully-connected layer.
fc_size = 128  # Number of neurons in fully-connected layer.


def initiate():
	data.test.images, data.test.labels = load_data(data_directory, file_name_identifier, image_size_flat)
	data.test = data.test.init()
	data.test._name = "test"
	data.test.cls = np.argmax(data.test.labels, axis=1)


	x = tf.placeholder(tf.float32, shape=[None, image_size_flat], name='x')
	x_image = tf.reshape(x, [-1, image_size, image_size, num_channels])
	y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
	y_true_cls = tf.argmax(y_true, axis=1)

	layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels,
												filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)

	layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1,
												filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

	layer_flat, num_features = flatten_layer(layer_conv2)

	layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

	layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

	y_pred = tf.nn.softmax(layer_fc2)
	y_pred_cls = tf.argmax(y_pred, axis=1)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
															   labels=y_true)
	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
	correct_prediction = tf.equal(y_pred_cls, y_true_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver()



	return data, x, x_image, y_true, y_true_cls, layer_conv1, layer_conv2, weights_conv1, weights_conv2, layer_flat, \
		   num_features, layer_fc1, layer_fc1, layer_fc2, y_pred, y_pred_cls, cost, optimizer, correct_prediction, \
		   accuracy, saver



'''
	you need to create the data object
	Load the data with the load_data function
	and then use the init to re-structure the data correctly.
'''

# initiates some variables



def next_batch(self, batch_size, shuffle=True):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		# Shuffle for the first epoch
		if self._epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self._num_examples)
			np.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._labels = self.labels[perm0]
		# Go to the next epoch
		if start + batch_size > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = self._num_examples - start
			images_rest_part = self._images[start:self._num_examples]
			labels_rest_part = self._labels[start:self._num_examples]
			# Shuffle the data
			if shuffle:
				perm = np.arange(self._num_examples)
				np.random.shuffle(perm)
				self._images = self.images[perm]
				self._labels = self.labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			end = self._index_in_epoch
			images_new_part = self._images[start:end]
			labels_new_part = self._labels[start:end]
			return1 = np.concatenate((images_rest_part, images_new_part), axis=0)
			return2 = np.concatenate((labels_rest_part, labels_new_part), axis=0)
			return return1, return2
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._images[start:end], self._labels[start:end]


def load_data(data_directory, file_name_identifier, img_size_flat, file_format="png"):
	# data_directory is the directoy where the data is located. if you are loading from the ./training_data/ folder then this should be /training_data
	# file_name_identifier is a unique identifier that should be present in only one of the classes filenames
	# img_size_flat is the product of image height timed by image width. E.g in a 100x100 picture it would be 100 * 100 = 10 000
	# NOTE: this only works for images which are a square with sides ABCD where A=B=C=D. E.g. 4 equal sides

	# load all files in path
	files = glob.glob(os.path.join(data_directory, "*.{}".format(file_format)))

	# init labels and images
	labels = []
	images = []

	for f in files:
		# read the image as a ndarray
		image = cv2.imread(f, 0)
		image = np.asarray(image, dtype="float32")
		# image = image.flatten()
		# add current image to image list
		images.append(image)
		# print("images = {}".format(type(images)))
		# convert filename to 0 or 1 based on if it contains kw or not

		label = 0
		if f.find(file_name_identifier) == -1:
			label = 1
		# add label to labels list
		labels.append(label)

	# shuffle both lists in the same order eg. x = [1, 2, 3], y = [1, 2, 3] ----> x = [3, 1, 2], y = [3, 1, 2]
	images, labels = shuffle(images, labels, random_state=0)
	# converts the images and labels to np_arrays for use with the tensorflow functions
	images = np.asarray(images)
	labels = np.asarray(labels)

	# reshapes the images to be of the correct shape
	images = images.reshape(-1, img_size_flat)
	return images, labels


def one_hot_encode(labels):
	# creates an array for the labels
	one_hot_labels = []
	for label in labels:
		if label == 0:
			one_hot_labels.append([0, 1])  # appends the labels array with a one hot coded label for 0
		else:
			one_hot_labels.append([1, 0])  # appends the labels array with a one hot coded label for 0
	one_hot_labels = np.asarray(one_hot_labels)  # converts the array to np_array
	return one_hot_labels


def graph():
	# Now load the model from file. The way TensorFlow
	# does this is confusing and requires several steps.

	# Create a new TensorFlow computational graph.
	graph = tf.get_default_graph()

	global x
	global y_true
	global session
	global optimizer
	global accuracy

	session = tf.Session()
	session.run(tf.global_variables_initializer())

	data, x, x_image, y_true, y_true_cls, layer_conv1, layer_conv2, weights_conv1, weights_conv2, layer_flat, \
	num_features, layer_fc1, layer_fc1, layer_fc2, y_pred, y_pred_cls, cost, optimizer, correct_prediction, \
	accuracy, saver = initiate()

	load_dir = 'resized/load/meta'
	load_path_meta = os.path.join(load_dir, 'best_validation.meta')
	load_path = os.path.join(load_dir, 'best_validation')

	new_saver = tf.train.import_meta_graph(load_path_meta)

	w1 = graph.get_tensor_by_name("x:0")
	w2 = graph.get_tensor_by_name("y_true:0")

	print("Session at {} has been restored successfully".format(load_path))

	return [w1, w2]

class style_data:

	class set:

		def tostring(self):
			return "labels = {}, images = {}, cls = {}".format(len(self._labels), len(self._images), len(self.cls))

		def tostring_long(self):
			return "labels = {}, images = {}, cls = {}".format(self._labels, self._images, self.cls)

		def images(self):
			return self._images

		def labels(self):
			return self._labels

		def num_examples(self):
			return self._num_examples

		def epochs_completed(self):
			return self._epochs_completed

		def init(self):
			self._num_examples = len(self.images)
			self.labels = one_hot_encode(self.labels)
			self._images = self.images
			self._labels = self.labels
			return self

		_index_in_epoch = 0
		_epochs_completed = 0
		_num_examples = 0
		_labels = np.array([])
		_images = np.array([])
		cls = []

	def tostring(self):
		return "[\n\ttrain = [{}], \n\ttest = [{}], \n\tvalidation = [{}]\n]".format(self.train.tostring(),
																					 self.test.tostring(),
																					 self.validation.tostring())

	def tostring_long(self):
		return "[\n\ttrain = [{}], \n\ttest = [{}], \n\tvalidation = [{}]\n]".format(self.train.tostring_long(),
																					 self.test.tostring_long(),
																					 self.validation.tostring_long())



	_file_names = ''
	train = set()
	test = set()
	validation = set()



