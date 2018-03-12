#version 9.0


# License (MIT)
# Copyright (c) 2016 by Magnus Erik Hvass Pedersen

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS 
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




#############################################################################################
####################											#############################
####################				  Setup					 #############################
####################											#############################
#############################################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob

# import random
# import shuffle
import gzip
import numpy
import cv2
import skimage
import tensorflow as tf
import math
import os
import sys
import time

from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import timedelta


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile








# from tensorflow.contrib.learn.python.learn.datasets import base






#############################################################################################
####################											#############################
####################				   Setup					#############################
####################											#############################
#############################################################################################

# disables cpu instruction warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# make numpy not print full arrays rather than
#[0,1,2 ... 97,98,99]
#[0,1,2 ... 97,98,99]
#	  [...]
#[0,1,2 ... 97,98,99]
#[0,1,2 ... 97,98,99]

# np.set_printoptions(threshold=np.inf)


########################
####	General	#####
########################

true = True
false = False
none = None

optimization_iterations = 1000 					# The amount of iterations for the optimization
print_regularity = 100  						# How often the training accuracy is printed during optimization
img_size = 100 									# size of the images
img_size_flat = img_size * img_size  			# Images are stored in one-dimensional arrays of this length.
img_shape = (img_size, img_size)  				# Tuple with height and width of images used to reshape arrays.
num_channels = 1  								# Number of color channels for the images: 1 channel for gray-scale.
num_classes = 2  								# Number of classes, one class for each of 10 digits.
plt_show = true  								# To show the plotted values set to true, to never plot anything set to false
file_name_identifier = "kw"  					# something distinguishable to tell the two images apart
batch_size = 256								# Split the test-set into smaller batches of this size. If crash due to low memory lower this one
train_batch_size = 64							# The size each training cycle gets split into. Split into smaller batches of this size. If crash due to low memory lower this one
train_data_directory = "resized/train"			# directory to load the train images
test_data_directory = "resized/test"			# directory to load the train images
validation_data_directory = "resized/validate"	# directory to load the train images

########################
####	Layers	#####
########################

# Convolutional Layer 1.
filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 16  # There are 16 of these filters.
# Convolutional Layer 2.
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 36  # There are 36 of these filters.
# Fully-connected layer.
fc_size = 128  # Number of neurons in fully-connected layer.


#############################################################################################
####################											#############################
####################				 Functions				  #############################
####################											#############################
#############################################################################################

def plot_image(image, name=""):
	fig = plt.figure(0)
	fig.canvas.set_window_title(name)
	plt.imshow(image.reshape(img_shape),
			   interpolation='nearest',
			   cmap='gray')

	# Sets the title of the figure window
	if plt_show == True:
		plt.show()


def plot_conv_layer(layer, image, name=""):
	# Assume layer is a TensorFlow op that outputs a 4-dim tensor
	# which is the output of a convolutional layer,
	# e.g. layer_conv1 or layer_conv2.

	# Create a feed-dict containing just one image.
	# Note that we don't need to feed y_true because it is
	# not used in this calculation.
	feed_dict = {x: [image]}

	# Calculate and retrieve the output values of the layer
	# when inputting that image.
	values = session.run(layer, feed_dict=feed_dict)

	# Number of filters used in the conv. layer.
	num_filters = values.shape[3]

	# Number of grids to plot.
	# Rounded-up, square-root of the number of filters.
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a grid of sub-plots.
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot the output images of all the filters.
	for i, ax in enumerate(axes.flat):
		# Only plot the images for valid filters.
		if i < num_filters:
			# Get the output image of using the i'th filter.
			# See new_conv_layer() for details on the format
			# of this 4-dim tensor.
			img = values[0, :, :, i]

			# Plot image.
			ax.imshow(img, interpolation='nearest', cmap='gray')

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	# Sets the title of the figure window
	fig.canvas.set_window_title(name)
	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	if plt_show == True:
		plt.show()


def plot_conv_weights(weights, input_channel=0, name=""):
	# Assume weights are TensorFlow ops for 4-dim variables
	# e.g. weights_conv1 or weights_conv2.

	# Retrieve the values of the weight-variables from TensorFlow.
	# A feed-dict is not necessary because nothing is calculated.
	w = session.run(weights)

	# Get the lowest and highest values for the weights.
	# This is used to correct the colour intensity across
	# the images so they can be compared with each other.
	w_min = np.min(w)
	w_max = np.max(w)

	# Number of filters used in the conv. layer.
	num_filters = w.shape[3]

	# Number of grids to plot.
	# Rounded-up, square-root of the number of filters.
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a grid of sub-plots.
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot all the filter-weights.
	for i, ax in enumerate(axes.flat):
		# Only plot the valid filter-weights.
		if i < num_filters:
			# Get the weights for the i'th filter of the input channel.
			# See new_conv_layer() for details on the format
			# of this 4-dim tensor.
			img = w[:, :, input_channel, i]

			# Plot image.
			ax.imshow(img, vmin=w_min, vmax=w_max,
					  interpolation='nearest', cmap='seismic')

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	# Sets the title of the figure window
	fig.canvas.set_window_title(name)
	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	if plt_show == True:
		plt.show()


def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False, name=""):
	# Number of images in the test-set.
	num_test = len(data.test.images)

	# Allocate an array for the predicted classes which
	# will be calculated in batches and filled into this array.
	cls_pred = np.zeros(shape=num_test, dtype=np.int)
	

	

	# Now calculate the predicted classes for the batches.
	# We will just iterate through all the batches.
	# There might be a more clever and Pythonic way of doing this.

	# The starting index for the next batch is denoted i.
	i = 0

	while i < num_test:
		# The ending index for the next batch is denoted j.
		j = min(i + batch_size, num_test)

		# Get the images from the test-set between index i and j.
		images = data.test.images[i:j, :]

		# Get the associated labels.
		labels = data.test.labels[i:j, :]

		# Create a feed-dict with these images and labels.
		feed_dict = {x: images,
					 y_true: labels}

					 
		# print("i = {} ".format( i ))
		# print("j = {} ".format( j ))
		# print("batch_size = {} ".format( batch_size ))
		# print("num_test = {} ".format( num_test ))
		# print("len(data.test.images) = {} ".format( len(data.test.images) ))
		# print("len(feed_dict) = {} ".format( len(cls_pred) ))
		# print("len(images) = {} ".format( len(cls_pred) ))
		# print("len(labels) = {} ".format( len(cls_pred) ))
		# print("\n")
		
		#																				mnist		|	KW
		# print("data.test.images.shape = {} ".format( (data.test.images.shape) ))
		# print("\n")
		# print("len(feed_dict) = {} ".format( len(feed_dict) )) #(10000, )			# 2				|	2
		# print("data.test.images.shape = {} ".format( (data.test.images.shape) )) 	# 10000, 784	|	7000, 1000	|	total_pxl, img_size_flat
		# print("data.test.labels.shape = {} ".format( (data.test.labels.shape) )) 	# 10000, 784	|	7000, 2 	|	total_pxl, ?classes?
		# print("cls_pred.shape = {} ".format( (cls_pred.shape) )) 					# 10000, 		|	7000, 		|	
		# print("images.shape = {} ".format( (images.shape) )) 						# 256,784 		|	256, 100	|	batch_size, classes
		# print("labels.shape = {} ".format( (labels.shape) )) 						# 256,10 		|	70, 2		|	batch size, classes
		# print("y_pred_cls.shape = {} ".format( (y_pred_cls.shape) )) 				# ?, 			|	?, 			|	
		# print("test_pred.shape = {} ".format( (y_pred_cls.shape) )) 				# ?, 			|	?, 			|	
		# print("\n")
		
		# Calculate the predicted class using TensorFlow.
		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

		# Set the start-index for the next batch to the
		# end-index of the current batch.
		i = j

	# Convenience variable for the true class-numbers of the test-set.
	cls_true = data.test.cls

	# Create a boolean array whether each image is correctly classified.
	correct = (cls_true == cls_pred)

	# Calculate the number of correctly classified images.
	# When summing a boolean array, False means 0 and True means 1.
	correct_sum = correct.sum()

	# Classification accuracy is the number of correctly classified
	# images divided by the total number of images in the test-set.
	acc = float(correct_sum) / num_test

	# Print the accuracy.
	msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
	print(msg.format(acc, correct_sum, num_test))

	# Plot some examples of mis-classifications, if desired.
	if show_example_errors:
		print("Example errors:")
		plot_example_errors(cls_pred=cls_pred, correct=correct)

	# Plot the confusion matrix, if desired.
	if show_confusion_matrix:
		print("Confusion Matrix:")
		plot_confusion_matrix(cls_pred=cls_pred, name=name)


def plot_confusion_matrix(cls_pred, name=""):
	# This is called from print_test_accuracy().

	# cls_pred is an array of the predicted class-number for
	# all images in the test-set.

	# Get the true classifications for the test-set.
	cls_true = data.test.cls

	# Get the confusion matrix using sklearn.
	cm = confusion_matrix(y_true=cls_true,
						  y_pred=cls_pred)

	# Print the confusion matrix as text.
	print(cm)

	# Plot the confusion matrix as an image.
	plt.matshow(cm)

	# Make various adjustments to the plot.
	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes))
	plt.yticks(tick_marks, range(num_classes))
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title('test', y=1.08, fontsize=20)

	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	if plt_show == True:
		plt.show()


def plot_example_errors(cls_pred, correct):
	# This function is called from print_test_accuracy() below.

	# cls_pred is an array of the predicted class-number for
	# all images in the test-set.

	# correct is a boolean array whether the predicted class
	# is equal to the true class for each image in the test-set.

	# Negate the boolean array.
	incorrect = (correct == False)

	# Get the images from the test-set that have been
	# incorrectly classified.
	images = data.test.images[incorrect]

	# Get the predicted classes for those images.
	cls_pred = cls_pred[incorrect]

	# Get the true classes for those images.
	cls_true = data.test.cls[incorrect]

	# Plot the first 9 images.
	plot_images(images=images[0:9],
				cls_true=cls_true[0:9],
				cls_pred=cls_pred[0:9], name="Predicted vs Actual")


# Counter for total number of iterations performed so far.
total_iterations = 0
def optimize(num_iterations):
	# Ensure we update the global variable rather than a local copy.
	global total_iterations

	# Start-time used for printing time-usage below.
	start_time = time.time()

	for i in range(total_iterations,
				   total_iterations + num_iterations):

		# Get a batch of training examples.
		# batch_images now holds a batch of images and
		# batch_labels are the true labels for those images.





		batch_images, batch_labels = next_batch(data.train, train_batch_size)

		# print("batch_images = {}".format(batch_images))
		# print("batch_labels = {}".format(batch_labels))
		# print("y_true = {}".format(y_true))
        #
		# print("batch_images.shape = {}".format(batch_images.shape))
		# print("batch_labels.shape = {}".format(batch_labels.shape))
		# print("y_true.shape = {}".format(y_true.shape))


		# Put the batch into a dict with the proper names
		# for placeholder variables in the TensorFlow graph.
		feed_dict_train = {x: batch_images,
						   y_true: batch_labels}
		#

		
		
		# Run the optimizer using this batch of training data.
		# TensorFlow assigns the variables in feed_dict_train
		# to the placeholder variables and then runs the optimizer.
		session.run(optimizer, feed_dict=feed_dict_train)

		# Print status every 100 iterations.
		if i % print_regularity == 0:
			# Calculate the accuracy on the training-set.
			acc = session.run(accuracy, feed_dict=feed_dict_train)

			# Message for printing.
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

			# Print it.
			print(msg.format(i + 1, acc))

	# Update the total number of iterations performed.
	total_iterations += num_iterations

	# Ending time.
	end_time = time.time()

	# Difference between start and end-times.
	time_dif = end_time - start_time

	# Print the time-usage.
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):  # Use Rectified Linear Unit (ReLU)?
	# input = The previous layer.
	# num_inputs = Num. inputs from prev. layer.
	# num_outputs = Num. outputs.

	# Create new weights and biases.
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)

	# Calculate the layer as the matrix multiplication of
	# the input and weights, and then add the bias-values.
	layer = tf.matmul(input, weights) + biases

	# Use ReLU?
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer

def flatten_layer(layer):
	# Get the shape of the input layer.
	layer_shape = layer.get_shape()

	# The shape of the input layer is assumed to be:
	# layer_shape == [num_images, img_height, img_width, num_channels]

	# The number of features is: img_height * img_width * num_channels
	# We can use a function from TensorFlow to calculate this.
	num_features = layer_shape[1:4].num_elements()

	# Reshape the layer to [num_images, num_features].
	# Note that we just set the size of the second dimension
	# to num_features and the size of the first dimension to -1
	# which means the size in that dimension is calculated
	# so the total size of the tensor is unchanged from the reshaping.
	layer_flat = tf.reshape(layer, [-1, num_features])

	# The shape of the flattened layer is now:
	# [num_images, img_height * img_width * num_channels]

	# Return both the flattened layer and the number of features.
	return layer_flat, num_features

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):  # Use 2x2 max-pooling.
	# input = The previous layer.
	# num_input_channels = Num. channels in prev. layer.
	# filter_size = Width and height of each filter.
	# num_filters = Number of filters.

	# Shape of the filter-weights for the convolution.
	# This format is determined by the TensorFlow API.
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	# Create new weights aka. filters with the given shape.
	weights = new_weights(shape=shape)

	# Create new biases, one for each filter.
	biases = new_biases(length=num_filters)

	# Create the TensorFlow operation for convolution.
	# Note the strides are set to 1 in all dimensions.
	# The first and last stride must always be 1,
	# because the first is for the image-number and
	# the last is for the input-channel.
	# But e.g. strides=[1, 2, 2, 1] would mean that the filter
	# is moved 2 pixels across the x- and y-axis of the image.
	# The padding is set to 'SAME' which means the input image
	# is padded with zeroes so the size of the output is the same.
	layer = tf.nn.conv2d(input=input,
						 filter=weights,
						 strides=[1, 1, 1, 1],
						 padding='SAME')

	# Add the biases to the results of the convolution.
	# A bias-value is added to each filter-channel.
	layer += biases

	# Use pooling to down-sample the image resolution?
	if use_pooling:
		# This is 2x2 max-pooling, which means that we
		# consider 2x2 windows and select the largest value
		# in each window. Then we move 2 pixels to the next window.
		layer = tf.nn.max_pool(value=layer,
							   ksize=[1, 2, 2, 1],
							   strides=[1, 2, 2, 1],
							   padding='SAME')

	# Rectified Linear Unit (ReLU).
	# It calculates max(x, 0) for each input pixel x.
	# This adds some non-linearity to the formula and allows us
	# to learn more complicated functions.
	layer = tf.nn.relu(layer)

	# Note that ReLU is normally executed before the pooling,
	# but since relu(max_pool(x)) == max_pool(relu(x)) we can
	# save 75% of the relu-operations by max-pooling first.

	# We return both the resulting layer and the filter-weights
	# because we will plot the weights later.
	return layer, weights

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def plot_images(images, cls_true, cls_pred=None, name=""):
	assert len(images) == len(cls_true) == 9

	# Create figure with 3x3 sub-plots.
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Plot image.
		ax.imshow(images[i].reshape(100,100), cmap='gray')

		# Show true and predicted classes.
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		# Show the classes as the label on the x-axis.
		ax.set_xlabel(xlabel)

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	# Sets the title of the figure window
	fig.canvas.set_window_title(name)
	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	if plt_show==True:
		plt.show()

def load_data(data_directory):
	# load all files in path
	files = glob.glob(os.path.join(data_directory, "*.png"))

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
	images = np.asarray(images)
	labels = np.asarray(labels)
	
	
	
	images = images.reshape(-1,img_size_flat)
	# labels = labels.reshape(-1,2)
	
	# images = images.flatten('F')
	
	return images, labels

def exit():
	os._exit(-1)
	return


def one_hot_encode(labels):
	one_hot_labels = []
	print("labels.size = {}".format(len(labels)))
	for label in labels:
		if label == 0:
			one_hot_labels.append([0, 1])
		else:
			one_hot_labels.append([1, 0])
	one_hot_labels = np.asarray(one_hot_labels)
	print("one_hot_labels.size = {}".format(len(one_hot_labels)))
	return one_hot_labels


def test(images, labels, dtype=dtypes.float32, seed=None):
	seed1, seed2 = random_seed.get_seed(seed)
	# If op level seed is not set, use whatever graph level seed is returned
	np.random.seed(seed1 if seed is None else seed2)
	dtype = dtypes.as_dtype(dtype).base_dtype
	if dtype not in (dtypes.uint8, dtypes.float32):
		raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
						dtype)
	assert images.shape[0] == labels.shape[0], (
			'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
	_num_examples = images.shape[0]

	# Convert shape from [num examples, rows, columns, depth]
	# to [num examples, rows*columns] (assuming depth == 1)
	if dtype == dtypes.float32:
		# Convert from [0, 255] -> [0.0, 1.0].
		images = images.astype(np.float32)
		images = np.multiply(images, 1.0 / 255.0)
	return images, labels, _num_examples



#############################################################################################
####################											#############################
####################			  	 classes					#############################
####################											#############################
#############################################################################################

# from .data import next_batch
# from .data import data
# from .data import *

try:
	from .data import *
except Exception: #ImportError
	from data import *

#############################################################################################
####################											#############################
####################			   Instructions					#############################
####################											#############################
#############################################################################################








data.train.images, data.train.labels = load_data(train_data_directory)
data.test.images, data.test.labels = load_data(test_data_directory)
data.validation.images, data.validation.labels = load_data(validation_data_directory)

# print("\n")
# print("################################################")
#
# print("data.test.images[0] = {} ".format( len(data.test.images) ))
#
# print("################################################")
# print("\n")



data.train._num_examples = len(data.train.images)
data.test._num_examples = len(data.test.images)
data.validation._num_examples = len(data.validation.images)




#
# data.train.images, data.train.labels, data.train._num_examples = test(data.train.images, data.train.labels)
# data.test.images, data.test.labels, data.test._num_examples = test(data.train.images, data.train.labels)
# data.validation.images, data.validation.labels, data.validation._num_examples = test(data.train.images, data.train.labels)

data.train.num_examples = len(data.train.images)


# 0 for KW
# 1 for Random



data.train.labels = one_hot_encode(data.train.labels)
data.test.labels = one_hot_encode(data.test.labels)
data.validation.labels = one_hot_encode(data.validation.labels)


# print("data.test.images.shape = {} ".format( (data.test.images.shape) ))
# print("data.test.labels.shape = {} ".format( (data.test.labels.shape) ))
# print("\n")
# print("data.test.images.shape = {}".format(data.test.images.shape))
# print("data.test._images.shape = {}".format(data.test._images.shape))
# print("data.test.labels.shape = {}".format(data.test.labels.shape))
# print("data.test._labels.shape = {}".format(data.test._labels.shape))
# print("\n")

data.train._images = data.train.images
data.test._images = data.test.images
data.validation._images = data.validation.images

data.train._labels = data.train.labels
data.test._labels = data.test.labels
data.validation._labels = data.validation.labels

data.train._name = "train"
data.test._name = "test"
data.validation._name = "validation"


# data.train.images = data.train.images.reshape(len(data.train.images), None)

# print("data.test.images.shape = {}".format(data.test.images.shape))
# print("data.test._images.shape = {}".format(data.test._images.shape))
# print("data.test.labels.shape = {}".format(data.test.labels.shape))
# print("data.test._labels.shape = {}".format(data.test._labels.shape))


# print("\n")
# print("Size of:")
# print("- Training-set:\t\t{}".format(len(data.train.labels)))
# print("- Test-set:\t\t\t{}".format(len(data.test.labels)))
# print("- Validation-set:\t{}".format(len(data.validation.labels)))
# print("\n")
# print("data.test.images are of type = {}".format(type(data.test.images)))
# print("data.test.images[1] are of type = {}".format(type(data.test.images[1])))
# print("data.test.images[1][1] are of type = {}".format(type(data.test.images[1][1])))
# print("\n")
# print("data.test.labels are of type = {}".format(type(data.test.labels)))
# print("data.test.labels[1] are of type = {}".format(type(data.test.labels[1])))
# print("data.test.labels[1][1] are of type = {}".format(type(data.test.labels[1][1])))
# print("\n")

# print("data = {}".format(data.tostring(data)))
# print("labels = {}".format(data.test.labels))
# print("images = {}".format(data.test.images))



# session = tf.Session()
# session.run(images)
# session.close()


data.test.cls = np.argmax(data.test.labels, axis=1)



# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels
# plot_images(images=images, cls_true=cls_true, name="The 9 first images from the data") # xxx



#############################################################################################
####################											#############################
####################			  Layers stuff				  #############################
####################											#############################
#############################################################################################

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = \
	new_conv_layer(input=x_image,
				   num_input_channels=num_channels,
				   filter_size=filter_size1,
				   num_filters=num_filters1,
				   use_pooling=True)

layer_conv2, weights_conv2 = \
	new_conv_layer(input=layer_conv1,
				   num_input_channels=num_filters1,
				   filter_size=filter_size2,
				   num_filters=num_filters2,
				   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,
						 num_inputs=num_features,
						 num_outputs=fc_size,
						 use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
						 num_inputs=fc_size,
						 num_outputs=num_classes,
						 use_relu=False)

# The ? symbol is to signify None as we put in above
print(layer_conv1)  # should output <tf.Tensor 'Relu:0' shape=(?, 14, 14, 16) dtype=float32>
print(layer_conv2)  # should output <tf.Tensor 'Relu_1:0' shape=(?, 7, 7, 36) dtype=float32>
print(layer_flat)  # should output <tf.Tensor 'Reshape_1:0' shape=(?, 1764) dtype=float32>
print(num_features)  # should output 1764
print(layer_fc1)  # should output <tf.Tensor 'Relu_2:0' shape=(?, 128) dtype=float32>
print(layer_fc2)  # should output <tf.Tensor 'add_3:0' shape=(?, 10) dtype=float32>
print("\n")

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
														   labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#############################################################################################
####################											#############################
####################			   Session stuff				#############################
####################											#############################
#############################################################################################

session = tf.Session()
session.run(tf.global_variables_initializer())


# Prints accuracy before optimization
print_test_accuracy(show_example_errors=True, name="Example errors")
# Optimizes for num_iterations iterations
optimize(num_iterations=optimization_iterations)

# prints accuracy after optimization plus example errors and confusion matshow
print_test_accuracy(show_example_errors=True,
					show_confusion_matrix=True, name="Example errors")

#############################################################################################
####################											#############################
####################					Plotting				#############################
####################											#############################
#############################################################################################

# NOTE: Negative weights in plotted weights are shown with blue and positive weights with red

image1 = data.test.images[0]
image2 = data.test.images[13]

plot_image(image1, name="A random image from the test set, will be refereed to as image1")
plot_image(image2, name="A random image from the test set, will be refereed to as image2")

plot_conv_weights(weights=weights_conv1, name="Filter-weights for the first convolutional layer")

plot_conv_layer(layer=layer_conv1, image=image1, name="Filter-weights from layer 1 applied to image1")
plot_conv_layer(layer=layer_conv1, image=image2, name="Filter-weights from layer 1 applied to image2")

plot_conv_weights(weights=weights_conv2, input_channel=0,
				  name="Filter-weights for the second convolutional, channel 1 of 36")
plot_conv_weights(weights=weights_conv2, input_channel=1,
				  name="Filter-weights for the second convolutional, channel 2 of 36")

plot_conv_layer(layer=layer_conv2, image=image1, name="Filter-weights from layer 2 applied to image1")
plot_conv_layer(layer=layer_conv2, image=image2, name="Filter-weights from layer 2 applied to image1")

session.close()



