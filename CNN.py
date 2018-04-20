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
from sklearn.metrics import confusion_matrix
from datetime import timedelta

import tensorflow as tf

import warnings
import time

#local imports
import utils as utils
from data import *
import augment as aug

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
print_full_np_array = False

if print_full_np_array == True:
	np.set_printoptions(threshold=np.inf)


########################
####	General	#####
########################

true = True
false = False
none = None

image_size              = 200 	# size of the images
num_channels            = 1  	# Number of color channels for the images: 1 channel for gray-scale, 3 for color
num_augment             = 40     # How many augmentations to make each image into
filter_size1            = 10    # Layer 1. Convolution filters are filter_size x filter_size pixels. might change this to 0.178 * img_size xxx
num_filters1            = 16    # Layer 1. There are n of these filters.
filter_size2            = 10    # Layer 2. Convolution filters are n x n pixels.
num_filters2            = 36    # Layer 2. There are n of these filters.
fc_size                 = 128   # Number of neurons in fully-connected layer.
optimization_iterations = 100   # The amount of iterations for the optimization


print_and_save_regularity = 10                  # How often the accuracy is printed during optimization. Saves happen in same loop
image_size_flat = image_size * image_size  	    # Images are stored in one-dimensional arrays of this length.
image_shape = (image_size, image_size)  		# Tuple with height and width of images used to reshape arrays.
num_classes = 2  								# Number of classes, one class for each of 10 digits.
plt_show = False  								# To show the plotted values set to true, to never plot anything set to false
class_zero = "rand"
class_one = "kw"
file_name_identifier = "kw"  					# something distinguishable to tell the two images apart
batch_size = 256								# Split the test-set into smaller batches of this size. If crash due to low memory lower this one
train_batch_size = 64							# The size each training cycle gets split into. Split into smaller batches of this size. If crash due to low memory lower this one
train_data_directory = "resized/train"			# directory to load the train images
test_data_directory = "resized/test"			# directory to load the train images
validation_data_directory = "resized/validate"	# directory to load the train images
augment = True                                  # Whether or not to augment



x = None
y_true = None
session = None
optimizer = None
accuracy = None

#############################################################################################
####################											#############################
####################				 Functions			    	#############################
####################											#############################
#############################################################################################

# Counter for total number of iterations performed so far.
total_iterations = 0
global_best = 0
def optimize(num_iterations, data, saver):
	# Ensure we update the global variable rather than a local copy.
	global total_iterations
	global x
	global y_true
	global session
	global optimizer
	global accuracy
	global global_best

	# Start-time used for printing time-usage below.
	start_time = time.time()

	for i in range(total_iterations,
				   total_iterations + num_iterations):

		# Get a batch of training examples.
		# batch_images now holds a batch of images and
		# batch_labels are the true labels for those images.
		batch_images, batch_labels = next_batch(data.train, train_batch_size)


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
		if i % print_and_save_regularity == 0:
			# Calculate the accuracy on the training-set.
			acc = session.run(accuracy, feed_dict=feed_dict_train)

			# Message for printing.
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

			# Print it.
			print(msg.format(i + 1, acc))

			if acc >= global_best:
				save(saver, session)
				global_best = acc

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


def exit(msg="Program exited as expected with exit function", exit_code=-1):
	warnings.warn(msg)
	os._exit(exit_code)
	return


def initiate():

	print("Preparing data")
	if augment:
		aug.prepare_data(train_data_directory, file_name_identifier, image_shape, num_channels, num_augment)
		aug.prepare_data(test_data_directory, file_name_identifier, image_shape, num_channels, num_augment)
		aug.prepare_data(validation_data_directory, file_name_identifier, image_shape, num_channels, num_augment)


	print("Loading data")
	data.train.images, data.train.labels = load_data(train_data_directory, file_name_identifier, image_size_flat, num_channels)
	data.test.images, data.test.labels = load_data(test_data_directory, file_name_identifier, image_size_flat, num_channels)
	data.validation.images, data.validation.labels = load_data(validation_data_directory, file_name_identifier, image_size_flat, num_channels)

	print("Initiating data")
	data.train = data.train.init()
	data.test = data.test.init()
	data.validation = data.validation.init()


	data.train._name = "train"
	data.test._name = "test"
	data.validation._name = "validation"


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


def save(saver, session):
	save_dir = 'checkpoints/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_path = os.path.join(save_dir, 'best_validation')
	saver.save(sess=session, save_path=save_path)
	print("Model saved in path: %s" % save_path)


def load(saver, session):
	save_dir = 'checkpoints/'
	save_path = os.path.join(save_dir, 'best_validation')
	saver.restore(sess=session, save_path=save_path)
	print("Model restored from path: %s" % save_path)




#############################################################################################
####################											#############################
####################			   Instructions					#############################
####################											#############################
#############################################################################################

def main():
	# ensure we use the global variuables rather than local variables
	global x
	global y_true
	global session
	global optimizer
	global accuracy

	data, x, x_image, y_true, y_true_cls, layer_conv1, layer_conv2, weights_conv1, weights_conv2, layer_flat, \
	num_features, layer_fc1, layer_fc1, layer_fc2, y_pred, y_pred_cls, cost, optimizer, correct_prediction, \
	accuracy, saver = initiate()

	# Get the first images from the test-set.
	images = data.test.images[0:9]

	# Get the true classes for those images.
	cls_true = data.test.cls[0:9]

	# Plot the images and labels
	utils.plot_nine_images(images, class_one, class_zero, cls_true, plt_show, name="The 9 first images from the data")

	#############################################################################################
	####################											#############################
	####################			   Session stuff				#############################
	####################											#############################
	#############################################################################################

	session = tf.Session()
	session.run(tf.global_variables_initializer())

	# Prints accuracy before optimization
	utils.print_test_accuracy(data, batch_size, x, y_true, session, y_pred_cls, class_one, class_zero, plt_show, show_example_errors=True, name="Predicted vs Actual")
	# Optimizes for num_iterations iterations
	optimize(optimization_iterations, data, saver)

	# prints accuracy after optimization plus example errors and confusion matshow
	utils.print_test_accuracy(data, batch_size, x, y_true, session, y_pred_cls, class_one, class_zero, plt_show, confusion_matrix, show_example_errors=True, show_confusion_matrix=True, name="Predicted vs Actual")

	#############################################################################################
	####################											#############################
	####################			   saving the model				#############################
	####################											#############################
	#############################################################################################

	save(saver, session)

	#############################################################################################
	####################											#############################
	####################					Plotting				#############################
	####################											#############################
	#############################################################################################

	# NOTE: Negative weights in plotted weights are shown with blue and positive weights with red

	image1 = data.test.images[0]
	image2 = data.test.images[13]

	utils.plot_image(image1, image_shape, plt_show, name="A random image from the test set, will be refereed to as image1")
	utils.plot_image(image2, image_shape, plt_show, name="A random image from the test set, will be refereed to as image2")

	utils.plot_conv_weights(weights_conv1, session, plt_show, name="Filter-weights for the first convolutional layer")

	utils.plot_conv_layer(layer_conv1, image1, session, plt_show, x, name="Filter-weights from layer 1 applied to image1")
	utils.plot_conv_layer(layer_conv1, image2, session, plt_show, x, name="Filter-weights from layer 1 applied to image2")

	utils.plot_conv_weights(weights_conv2, session, plt_show, convolutional_layer=0, name="Filter-weights for the second convolutional, channel 1 of 36")
	utils.plot_conv_weights(weights_conv2, session, plt_show, convolutional_layer=1, name="Filter-weights for the second convolutional, channel 2 of 36")

	utils.plot_conv_layer(layer_conv2, image1, session, plt_show, x, name="Filter-weights from layer 2 applied to image1")
	utils.plot_conv_layer(layer_conv2, image2, session, plt_show, x, name="Filter-weights from layer 2 applied to image1")

	session.close()


if __name__ == "__main__":
	# stuff only to run when not called via 'import' here
	main()
