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
# import numpy
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



class data:
	class train:
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

		_index_in_epoch = 0
		_epochs_completed = 0
		_num_examples = 0
		_labels = np.array([])
		_images = np.array([])
		cls = []

	class test:
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

		_name = ""

		_index_in_epoch = 0
		_epochs_completed = 0
		_num_examples = 0
		_labels = np.array([])
		_images = np.array([])
		cls = []

	class validation:
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
	train = train()
	test = test()
	validation = validation()





def next_batch(self, batch_size, shuffle=True):
		"""Return the next `batch_size` examples from this data set."""

		# print("\n")
		# print("labels.shape = {}".format(self.labels.shape))
		# print("images.shape = {}".format(self.images.shape))
		# print("_labels.shape = {}".format(self._labels.shape))
		# print("_images.shape = {}".format(self._images.shape))
		# print("\n")
		# print("self._index_in_epoch = {}".format(self._index_in_epoch))
		# print("self._epochs_completed = {}".format(self._epochs_completed))
		# print("self._num_examples = {}".format(self._num_examples))
		# print("\n")



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

			# print("first return")
			return return1, return2
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch

			#reshaping the labels
			# because the array is structured [0,1,1,0,0,1,1,0,1,0,1,0] which will be [[0,1],[1,0],[0,1],[1,0],[1,0],[1,0]]
			# so if you batch size is 64, you need 128 integers to make the output a (64, 2) shape array
			# printer = start
			# helper = start+(batch_size*2)
			# _label_reshaper = self._labels[start:helper]
			# _label_reshaper = _label_reshaper.reshape(-1, 2)
			# tester = _label_reshaper.shape
			return self._images[start:end], self._labels[start:end]