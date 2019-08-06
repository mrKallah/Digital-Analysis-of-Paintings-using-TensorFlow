from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.utils import shuffle

import numpy as np

import glob
import cv2
import os

'''
	you need to create the data object
	Load the data with the load_data function
	and then use the init to re-structure the data correctly.
'''


def next_batch(self, batch_size, shuffle=True, test=False):
	"""
	Return the next `batch_size` examples from this data set.
	:param self: the data object
	:param batch_size: the size of the data to return
	:param shuffle: whether or not to shuffle the data
	:param test: used for testing, resets the _index_in_epoch
	:return: the next N number of images and labels
	"""
	if test == True:
		self._index_in_epoch = 0

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


def load_data(data_directory, img_size_flat, channels, file_format="png", augment=True):
	"""

	:param data_directory: the directoy where the data is located. if you are loading from the ./training_data/ folder then this should be a string of "/training_data"
	:param file_name_identifier: a unique identifier that should be present in only one of the classes file names
	:param img_size_flat: the product of image height timed by image width. E.g in a 100x100 picture it would be 100 * 100 = 10 000
			Note: this only works for images which are a square with sides ABCD where A=B=C=D. E.g. 4 equal sides
	:param channels: the number of color channels, 1 for greyscale and 3 for color
	:param file_format: the format the augmented images are. The default is png
	:param augment: Weather or not to read from the /augmented/ folder
	:return: the images and labels it has read.
	"""

	# data_directory is the directoy where the data is located. if you are loading from the ./training_data/ folder then this should be /training_data
	# file_name_identifier is a unique identifier that should be present in only one of the classes filenames
	# img_size_flat is the product of image height timed by image width. E.g in a 100x100 picture it would be 100 * 100 = 10 000
	# NOTE: this only works for images which are a square with sides ABCD where A=B=C=D. E.g. 4 equal sides

	# load all files in path
	if augment == True:
		files = glob.glob(os.path.join(data_directory, "augmented/", "*.{}".format(file_format)))
	else:
		files = glob.glob(os.path.join(data_directory, "*.{}".format(file_format)))

	# init labels and images
	labels = []
	images = []

	for f in files:
		# read the image as a ndarray
		if channels == 1:
			image = cv2.imread(f, 0)
		else:
			image = cv2.imread(f, channels)

		image = np.asarray(image, dtype="float32")

		# add current image to image list
		images.append(image)

		label = rm_path(f, file_format)


		# add label to labels list
		labels.append(label)

	# shuffle both lists in the same order eg. x = [1, 2, 3], y = [1, 2, 3] ----> x = [3, 1, 2], y = [3, 1, 2]
	images, labels = shuffle(images, labels, random_state=0)
	# converts the images and labels to np_arrays for use with the tensorflow functions
	images = np.asarray(images)
	labels = np.asarray(labels)

	# reshapes the images to be of the correct shape
	if channels == 3:
		images = images.reshape(-1, img_size_flat, 3)
	else:
		print(img_size_flat)
		print(images.shape)
		images = images.reshape(-1, img_size_flat)
	return images, labels


def rm_path(label, file_format):
	label = label.replace("resized/train/augmented/", "")
	label = label.replace("resized/validate/augmented/", "")
	label = label.replace("resized/test/augmented/", "")
	label = label.replace("resized/load/chosen/", "")
	label = label.replace("test_data/augmented/", "")
	label = label.replace(".png", "")
	label = label.replace(".jpg", "")
	label = label.replace(".jpeg", "")
	label = label.replace(file_format, "")
	label = label.replace("_", "")

	label = label.replace("0", "")
	label = label.replace("1", "")
	label = label.replace("2", "")
	label = label.replace("3", "")
	label = label.replace("4", "")
	label = label.replace("5", "")
	label = label.replace("6", "")
	label = label.replace("7", "")
	label = label.replace("8", "")
	label = label.replace("9", "")
	return label


def one_hot_encode(labels, classes, file_format):
	"""
	One hot encodes an array of two classes. This does not scale
	TODO: This needs to be inverted
	:param labels: the array of the classes
	:return: the encoded array
	"""
	# creates an array for the labels
	one_hot_labels = []
	num_classes = np.size(classes)
	j = 0
	for label in labels:
		new_labels = []
		label = rm_path(label, file_format)

		i = 0
		for c in classes:
			if label == c:
				label = i
			i += 1
		i = 0

		while (i < num_classes):
			new_labels.append(0)
			i = i + 1

		#sets label to 0, which should be random, if filename is unknown
		if label.__class__ == "".__class__:
			label = 0
		new_labels[label] = 1
		j = j + 1
		one_hot_labels.append(new_labels)

	one_hot_labels = np.asarray(one_hot_labels)
	return one_hot_labels


class data:
	class set:

		def tostring(self):
			"""
			A tostring that prints the shape of the labels, images and CLS
			:return: the shape of the labels, images and CLS structured in a string with labels
			"""
			return "labels = {}, \nimages = {}, \ncls = {}".format(len(self._labels.shape), len(self._images.shape), len(self.cls.shape))

		def tostring_long(self):
			"""
			A tostring that prints the labels, images and CLS
			:return: the labels, images and CLS structured in a string with labels
			"""
			return "labels = {}, \nimages = {}, \ncls = {}".format(self._labels, self._images, self.cls)

		def images(self):
			"""
			returns the images saved in the _images variable
			:return: an array of images
			"""
			return self._images

		def labels(self):
			"""
			returns the labels saved in the _labels variable
			:return: an array of labels
			"""
			return self._labels

		def num_examples(self):
			"""
			returns the number of images in the images array
			:return: the _num_examples variable
			"""
			return self._num_examples

		def epochs_completed(self):
			"""
			how many epochs the class has gone through
			:return: the _epochs_completed variable
			"""
			return self._epochs_completed

		def init(self, classes):
			"""
			Initiates the variables, program will not work if this is not called.
			:return: an initiated version of self
			"""
			self._num_examples = len(self.images)
			self.labels = one_hot_encode(self.labels, classes, file_format=".png")
			self._images = self.images
			self._labels = self.labels
			return self

		_index_in_epoch = 0
		_epochs_completed = 0
		_num_examples = 0
		_labels = np.array([])
		_images = np.array([])
		cls = []

	_file_names = ''
	train = set()
	test = set()
	validation = set()
