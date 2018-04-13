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


def load_data(data_directory, file_name_identifier, img_size_flat, channels, file_format="png", augment=True):
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


class data:
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
