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

# local imports
import Utils as utils
from Data import *
import Augment as aug

#############################################################################################
####################											#############################
####################				   Setup					#############################
####################											#############################
#############################################################################################

# disables cpu instruction warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

########################
####	General	#####
########################

true = True  # I am used to Java
false = False  # I am used to Java
none = None  # I am used to Java

image_size              = 75 	# size of the images
num_channels            = 3 	# Number of color channels for the images: 1 channel for gray-scale, 3 for color
num_augment             = 10 	# How many augmentations to make each image into (1,10,20,30,40)
filter_size1            = 10 	# Layer 1. Convolution filters are filter_size x filter_size pixels.
num_filters1            = 16	# Layer 1. There are n of these filters.
filter_size2            = 10	# Layer 2. Convolution filters are n x n pixels.
num_filters2            = 36	# Layer 2. There are n of these filters.
fc_size                 = 64	# Number of neurons in fully-connected layer.
optimization_iterations = 200000	# The amount of iterations for the optimization
overfit_avoidance		= 50	#How many times the accuracy of the model can be 100 before it stops the optimization. -1 to turn off

plt_show = True  # Whether or not to show the plots. Set to true to plot, set to false to avoid plotting. (increases performance and won’t stop the program, useful for run_many)
augment = False  # Whether or not to augment, if img_size or num_augment has change this must be true. Saves time not to re-augment every time you run the program if not needed.
print_and_save_regularity = 1  # How often the accuracy is printed during optimization. Saves happen in same loop


batch_size = 512  # This is used for how large the batches are when finding the test accuracy. Lowering this will use less ram but be slower.
train_batch_size = 64  # The size each training cycle gets split into. Split into smaller batches of this size. If crash due to low memory lower this one

classes = ["random", "gogh", "bernard", "bonnard", "ibels", "kunisada", "kw"]  #an array of the classes, 0 must be random
train_data_directory = "resized/train"  # directory to load the train images from
test_data_directory = "resized/test"  # directory to load the test images from
# validation_data_directory = "resized/validate"  # directory to load the validate images from

image_size_flat = image_size * image_size  # Images are stored in one-dimensional arrays of this length.
image_shape = (image_size, image_size)  # Tuple with height and width of images used to reshape arrays.
num_classes = np.size(classes) # Number of classes, one class for each of 10 digits.

file_name_identifier = ".png" #png is already included, change to other filename to add

print_full_np_array = False  # Make numpy not print full arrays rather than:
# [0,1,2 ... 97,98,99]
# [0,1,2 ... 97,98,99]
# 	    [...]
# [0,1,2 ... 97,98,99]
# [0,1,2 ... 97,98,99]
# This can cause a lot of delay as printing takes long

if print_full_np_array == True:
	np.set_printoptions(threshold=np.inf)


#############################################################################################
####################											#############################
####################				 Functions			    	#############################
####################											#############################
#############################################################################################
def crash_xming():
	"""
	Causes the program to crash if xming is not running, but required to run,
	This is for users using Bash on Ubuntu On Windows
	It's better to crash early, than after it spends 15 minutes augmenting the data
	"""
	import matplotlib.pyplot as plt
	fig = plt.figure()
	fig.canvas.set_window_title("")
	plt.suptitle("")
	img = cv2.imread("test_data/augmented/org.png", 3)
	plt.imshow(img) #xxx blue castle is from here
	plt.close()
	del plt


def init_rand_var(do_augment):
	"""
	Initiates global variables for the run_many program to change each run
	:param do_augment: weather or not to augment
	"""
	global total_iterations
	global global_best
	global test_accuracy
	global total_time
	global x
	global y_true
	global session
	global optimizer
	global accuracy
	global print_and_save_regularity
	global image_size_flat
	global image_shape
	global num_classes
	global plt_show
	global class_zero
	global class_one
	global file_name_identifier
	global batch_size
	global train_batch_size
	global train_data_directory
	global test_data_directory
	# global validation_data_directory
	global augment

	total_iterations = 0
	global_best = 0
	test_accuracy = 0
	total_time = 0
	x = None
	y_true = None
	session = None
	optimizer = None
	accuracy = None
	print_and_save_regularity = 1
	image_size_flat = image_size * image_size
	image_shape = (image_size, image_size)
	num_classes = 2
	plt_show = False
	class_zero = "rand"
	class_one = "gogh"
	file_name_identifier = "gogh"
	batch_size = 256  # 256
	train_batch_size = 64  # 64
	train_data_directory = "resized/train"
	test_data_directory = "resized/test"
	# validation_data_directory = "resized/validate"
	augment = do_augment


# Counter for total number of iterations performed so far.
total_iterations = 0
global_best = 0
fitting = 0
def optimize(num_iterations, data, saver):
	"""
	@author = Magnus Erik Hvass Pedersen
	Optimizes the network on a batch of data. The larger the batch, the better, however large batch sizes uses a lot of
	system resources
	Change the batch size with train_batch_size
	Saves the latest best model
	:param num_iterations: how many iterations to optimize.
	:param data: the data object
	:param saver: the TF saver object
	:return:
	"""
	# Ensure we update the global variable rather than a local copy.
	global total_iterations
	global x
	global y_true
	global session
	global optimizer
	global accuracy
	global global_best
	global overfit_avoidance
	global fitting

	# Start-time used for printing time-usage below.
	start_time = time.time()

	for i in range(total_iterations, total_iterations + num_iterations):

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

			# Print current accuracy and iteration
			print("Optimization Iteration: {0:>6}/{1}, Training Accuracy: {2:>6.1%}".format(i + 1, num_iterations, acc))

			if acc >= global_best:
				save(saver, session)
				global_best = acc
			if overfit_avoidance != -1:
				if acc == 1:
					fitting = fitting +1
				else:
					fitting = 0
			if fitting != 0:
				print("Over fitting protection = {}/{}".format(fitting, overfit_avoidance))
		if fitting == overfit_avoidance:
			return

	# Update the total number of iterations performed.
	total_iterations += num_iterations

	# Ending time.
	end_time = time.time()

	# Difference between start and end-times.
	time_dif = end_time - start_time

	global total_time
	global test_accuracy
	total_time = str(timedelta(seconds=int(round(time_dif))))

	# Print the time-usage.
	print("Time usage: " + total_time)


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	"""
	@author = Magnus Erik Hvass Pedersen
	creates a new fully connected layer
	:param input: The previous layer
	:param num_inputs: Number inputs from previous layer
	:param num_outputs: Number of outputs, should be the same as number of classes for most cases
	:param use_relu: boolean variable to say if you want to use ReLU, True to use ReLU, Default: True
	:return:
	"""

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
	"""
	@author = Magnus Erik Hvass Pedersen
	Flattens a layer
	:param layer: the layer to flatten
	:return: the new flattened layer and the number of features within it
	"""
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


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	"""
	@author = Magnus Erik Hvass Pedersen
	creates a new convolutional layer
	:param input: The previous layer
	:param num_input_channels: Number of channels in previous layer
	:param filter_size: Width and height of each filter
	:param num_filters: Number of filters
	:param use_pooling: Boolean variable to say if you want to use 2x2 max-pooling., True to use pooling, Default: True
	:return: the convolutional layer and the filter-weights. The weights are used for plotting
	"""

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
	"""
	@author = Magnus Erik Hvass Pedersen
	creating the biases for the network
	"""
	return tf.Variable(tf.constant(0.05, shape=[length]))


def new_weights(shape):
	"""
	creating the random weights for the network
	@author = Magnus Erik Hvass Pedersen
	"""
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def exit(msg="", exit_code=-1):
	"""
	Exits the program with a custom message.
	:param msg: the message to give the user on exit. Default: Program exited as expected with exit function
	:param exit_code: the exit code. can be any value between 0-255, default: -1 (255)
	"""
	if msg != "":
		warnings.warn(msg)
	os._exit(exit_code)


def write_settings_to_file():
	settings = [image_size, num_channels, num_augment, filter_size1,
				num_filters1, filter_size2, num_filters2, fc_size,
				optimization_iterations, num_classes]


	file = open("checkpoints/checkpoint.settings", "w")
	for setting in settings:
		file.write("{}\n".format(setting))
	file.close()

	return

def read_settings_from_file(file="checkpoints/checkpoint.settings"):
	settings = []

	file = open(file, "r")
	for f in file:
		for s in f.split():
			if s.isdigit():
				settings.append(s)

	image_size = int(settings[0])
	num_channels = int(settings[1])
	num_augment = int(settings[2])
	filter_size1 = int(settings[3])
	num_filters1 = int(settings[4])
	filter_size2 = int(settings[5])
	num_filters2 = int(settings[6])
	fc_size = int(settings[7])
	optimization_iterations = int(settings[8])
	num_classes = int(settings[9])
	
	# Tuple with height and width of images used to reshape arrays.
	if num_channels == 1:
		image_shape = (image_size, image_size)	
	else:
		image_shape = (image_size, image_size, num_channels)
		
		
	# Images are stored in one-dimensional arrays of this length.
	image_size_flat = image_size * image_size  	
	file.close()
	return image_size, num_channels, num_augment, filter_size1, num_filters1, filter_size2, num_filters2, fc_size, optimization_iterations, num_classes, image_size_flat, image_shape

def save(saver, session, save_dir="checkpoints/", file_name="best_validation"):
	"""
	Saves the session to a file
	:param saver: the tf saver object
	:param session: the session to save
	:param save_dir: where to save the model, default: "checkpoints/"
	:param file_name: the name to save the model files as, default: "best_validation"
	"""

	# creates the save directory
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# creates the name to save the files as
	save_path = os.path.join(save_dir, file_name)

	# saves the file to the disk
	saver.save(sess=session, save_path=save_path)

	write_settings_to_file()

	print("Model saved in path: %s" % save_path)


def load(saver, session, save_dir="checkpoints/", file_name="best_validation", read_from_file=False):
	"""
	Loads the session from file
	:param saver: the tf saver object
	:param session: the session to overwrite
	:param save_dir: where to save the model, default: "checkpoints/"
	:param file_name: the name to save the model files as, default: "best_validation"
	"""
	if read_from_file:
		read_settings_from_file()

	# Creates the path to load from
	save_path = os.path.join(save_dir, file_name)

	# restors the model from disk
	saver.restore(sess=session, save_path=save_path)

	print("Model restored from path: %s" % save_path)


def print_var():
	"""
	returns variables for printing results and parameters for the run_many program
	:return: the variables that are used for the data collection, and the results of the run
	"""
	# return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}" \
	# 	.format(image_size, num_channels, num_augment, filter_size1, num_filters1, filter_size2, num_filters2, \
	#             fc_size, optimization_iterations, test_accuracy, total_time, cm)

	return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}" \
		.format(image_size, num_channels, num_augment, filter_size1, num_filters1, filter_size2, num_filters2, \
	            fc_size, optimization_iterations, total_time)


def print_header():
	"""
	this is to print a header for the run_many program
	:return: a string of the identifiers for the print_var function
	"""
	return "image_size\tnum_channels\tnum_augment\tfilter_size1\tnum_filters1\tfilter_size2\tnum_filters2\tfc_size\toptimization_iterations\ttest_acc\ttime\tconfusion_matrix"


def set_parameters(img_size, num_chan, num_aug, fs1, num_fs1, fs2, num_fs2, size_fc, num_optim_iter):
	"""
	Sets all the parameters for the program
	:param img_size: size of the images
	:param num_chan: Number of color channels for the images: 1 channel for gray-scale, 3 for color
	:param num_aug: How many augmentations to make each image into
	:param fs1: Layer 1. Convolution filters are filter_size x filter_size pixels.
	:param num_fs1: Layer 1. There are n of these filters.
	:param fs2: Layer 2. Convolution filters are n x n pixels.
	:param num_fs2: Layer 2. There are n of these filters.
	:param size_fc: Number of neurons in fully-connected layer.
	:param num_optim_iter: The amount of iterations for the optimization
	"""
	# makes sure all the vars are global
	global image_size
	global num_channels
	global num_augment
	global filter_size1
	global num_filters1
	global filter_size2
	global num_filters2
	global fc_size
	global optimization_iterations

	# sets the global variables to the new ones from the function parameters
	image_size = img_size
	num_channels = num_chan
	num_augment = num_aug
	filter_size1 = fs1
	num_filters1 = num_fs1
	filter_size2 = fs2
	num_filters2 = num_fs2
	fc_size = size_fc
	optimization_iterations = num_optim_iter


def run_many(img_size, num_chan, num_aug, fs1, num_fs1, fs2, num_fs2, size_fc, num_optim_iter, do_augment=True):
	"""
	lets you run the program many times with different parameters each time
	:param img_size: size of the images
	:param num_chan: Number of color channels for the images: 1 channel for gray-scale, 3 for color
	:param num_aug: How many augmentations to make each image into
	:param fs1: Layer 1. Convolution filters are filter_size x filter_size pixels.
	:param num_fs1: Layer 1. There are n of these filters.
	:param fs2: Layer 2. Convolution filters are n x n pixels.
	:param num_fs2: Layer 2. There are n of these filters.
	:param size_fc: Number of neurons in fully-connected layer.
	:param num_optim_iter: The amount of iterations for the optimization
	:param do_augment: weather or not to augment the images, saves time if off when not needed. Default True
	"""

	# sets the parameters
	set_parameters(img_size, num_chan, num_aug, fs1, num_fs1, fs2, num_fs2, size_fc, num_optim_iter)

	# resets the variables that needs resetting
	init_rand_var(do_augment)

	# runs the program
	main()

	# writes the results to a file called result.txt
	f = open("Results.txt", "a+")
	f.write(print_var())
	f.write("\n")
	f.close()


def initiate():
	"""
	Function to set all the variables for the classifier to run
	:return: all the variables created.
	"""
	print("Preparing data")
	if augment:
		aug.prepare_data(train_data_directory, image_shape, num_channels, num_augment)
		aug.re_iterate()
		aug.prepare_data(test_data_directory, image_shape, num_channels, num_augment)
		aug.re_iterate()
		# aug.prepare_data(validation_data_directory, image_shape, num_channels, num_augment)
		# aug.re_iterate()

	print("Loading data")
	data.train.images, data.train.labels = load_data(train_data_directory, image_size_flat,
	                                                 num_channels)
	data.test.images, data.test.labels = load_data(test_data_directory, image_size_flat,
	                                               num_channels)
	# data.validation.images, data.validation.labels = load_data(validation_data_directory,
	#                                                            image_size_flat, num_channels)


	print("Initiating data")
	data.train = data.train.init(classes)
	data.test = data.test.init(classes)
	# data.validation = data.validation.init(classes)


	data.train._name = "train"
	data.test._name = "test"
	# data.validation._name = "validation"

	# The labels without one hot encoding
	data.test.cls = np.argmax(data.test.labels, axis=1)

	print("Creating TF placeholder objects- and variables, fully connected layers and convolutional layers")
	# this creates the tf placeholder object for the images.
	if num_channels == 3:
		x = tf.placeholder(tf.float32, shape=[None, image_size_flat, num_channels], name='x')
	else:
		x = tf.placeholder(tf.float32, shape=[None, image_size_flat], name='x')

	# A 4 dimentional tensor is needed for the convolutional layers, and so we have to reshape it to be
	# [-1, image_size, image_size, num_channels]
	# where -1 is the amount of images, and is inferred automatically by tensorflow, image_size which is the size of the
	# images, where this requires height=width, and num channels which is the color channels
	x_image = tf.reshape(x, [-1, image_size, image_size, num_channels])

	# This is the TF placeholder object for the labels and is related to the x placeholder
	y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

	# This is the true labels, in a tensorflow variable
	y_true_cls = tf.argmax(y_true, axis=1)

	# Here the first convolutional layer is created
	layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels,
	                                            filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)

	# Here the second convolutional layer is created
	layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1,
	                                            filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

	# This flattens the second convolutional layer
	layer_flat, num_features = flatten_layer(layer_conv2)

	# This creates the first fully connected layer, using ReLU
	layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

	# This creates the first fully connected layer, not using ReLU
	layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

	# The prediction of the all classes, normalized, if you have 70% certainty for class_one and 30% for class_two,
	# this will be something like [0.7, 0.3]
	y_pred = tf.nn.softmax(layer_fc2)

	# Using the y_pred variable, you take the max value, and saves it as one, and everything else as 0, so in previous
	# example of [0.7, 0.3], this would become [1, 0]
	y_pred_cls = tf.argmax(y_pred, axis=1)

	# this is used with back propagation to increase the accuracy of the network
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)

	# the average of the cross-entropy for all the image classifications.
	cost = tf.reduce_mean(cross_entropy)

	# creates an optimizer for later use using the adam optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

	#
	correct_prediction = tf.equal(y_pred_cls, y_true_cls)

	# a vector of booleans whether the predicted class equals the true class of each image.
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# this creates a saver, so the model can later be saved
	saver = tf.train.Saver()

	return data, x, x_image, y_true, y_true_cls, layer_conv1, layer_conv2, weights_conv1, weights_conv2, layer_flat, \
	       num_features, layer_fc1, layer_fc1, layer_fc2, y_pred, y_pred_cls, cost, optimizer, correct_prediction, \
	       accuracy, saver


def main():
	# ensure we use the global variables rather than local variables
	global x
	global y_true
	global session
	global optimizer
	global accuracy

	# sets all the variables needed to run the program
	data, x, x_image, y_true, y_true_cls, layer_conv1, layer_conv2, weights_conv1, weights_conv2, layer_flat, \
	num_features, layer_fc1, layer_fc1, layer_fc2, y_pred, y_pred_cls, cost, optimizer, correct_prediction, \
	accuracy, saver = initiate()

	# Get the first images from the test-set.
	images = data.test.images[0:9]

	# Get the true classes for those images.
	cls_true = data.test.cls[0:9]

	# Plot the images and labels
	utils.plot_nine_images(images, classes, cls_true, plt_show, image_shape, channels=num_channels,
	                       name="The 9 first images from the data")

	#############################################################################################
	####################											#############################
	####################			   Session stuff				#############################
	####################											#############################
	#############################################################################################

	print("Starting session")
	# Creates the session
	session = tf.Session()
	session.run(tf.global_variables_initializer())

	# Prints accuracy before optimization
	print("Calculating test accuracy")
	utils.print_test_accuracy(data, batch_size, x, y_true, session, y_pred_cls, classes, plt_show,
	                          channels=num_channels, img_shape=image_shape, show_example_errors=True,
	                          name="Predicted vs Actual")

	# Optimizes for num_iterations iterations
	print("Optimising for {} iterations".format(optimization_iterations))
	optimize(optimization_iterations, data, saver)

	# Loads the best session from the optimization
	load(saver, session)

	global test_accuracy
	global cm
	# Prints accuracy after optimization plus example errors and confusion matrix
	print("Printing test accuracy")

	test_accuracy, cm = utils.print_test_accuracy(data, batch_size, x, y_true, session, y_pred_cls, classes,
												  plt_show, channels=num_channels, confusion_matrix=confusion_matrix,
												  img_shape=image_shape, show_example_errors=True,
												  show_confusion_matrix=True, name="Predicted vs Actual")


	#############################################################################################
	####################											#############################
	####################					Plotting				#############################
	####################											#############################
	#############################################################################################

	# NOTE: Negative weights in plotted weights are shown with blue and positive weights with red

	image1 = data.test.images[0]
	image2 = data.test.images[13]

	if plt_show == true:
		utils.plot_image(image1, image_shape, plt_show, num_channels,
		                 name="A random image from the test set, will be refereed to as image1")

		utils.plot_image(image2, image_shape, plt_show, num_channels,
		                 name="A random image from the test set, will be refereed to as image2")

		utils.plot_conv_weights(weights_conv1, session, plt_show,
		                        name="Filter-weights for the first convolutional layer")

		utils.plot_conv_layer(layer_conv1, image1, session, plt_show, x,
		                      name="Filter-weights from layer 1 applied to image1")

		utils.plot_conv_layer(layer_conv1, image2, session, plt_show, x,
		                      name="Filter-weights from layer 1 applied to image2")

		utils.plot_conv_weights(weights_conv2, session, plt_show, convolutional_layer=0,
		                        name="Filter-weights for the second convolutional, channel 1 of 36")

		utils.plot_conv_weights(weights_conv2, session, plt_show, convolutional_layer=1,
		                        name="Filter-weights for the second convolutional, channel 2 of 36")

		utils.plot_conv_layer(layer_conv2, image1, session, plt_show, x,
		                      name="Filter-weights from layer 2 applied to image1")

		utils.plot_conv_layer(layer_conv2, image2, session, plt_show, x,
		                      name="Filter-weights from layer 2 applied to image1")
	# ends the session
	session.close()


if __name__ == "__main__":
	# stuff only to run when not called via 'import' here

	crash_xming()
	main()
