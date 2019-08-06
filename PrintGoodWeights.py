from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

#local imports
from Data import *
import Utils as utils
from Train import new_fc_layer
from Train import flatten_layer
from Train import new_conv_layer
import resized.load.resize_to_chosen as resize

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

true = True    # I am used to Java
false = False  # I am used to Java
none = None    # I am used to Java

image_size              = 125 	# size of the images
num_channels            = 3  	# Number of color channels for the images: 1 channel for gray-scale, 3 for color
filter_size1            = 4   # Layer 1. Convolution filters are filter_size x filter_size pixels.
num_filters1            = 20    # Layer 1. There are n of these filters.
filter_size2            = 5    # Layer 2. Convolution filters are n x n pixels.
num_filters2            = 64   # Layer 2. There are n of these filters.
fc_size                 = 512   # Number of neurons in fully-connected layer.
optimization_iterations = 600   # The amount of iterations for the optimization

batch_size = 256				                # Split the test-set into smaller batches of this size. If crash due to low memory lower this one
train_batch_size = 64			                # The size each training cycle gets split into. Split into smaller batches of this size. If crash due to low memory lower this one
print_and_save_regularity = 1                   # How often the accuracy is printed during optimization. Saves happen in same loop
class_zero = "Random"                             # the name of the random class, this is what it will be printed as
class_one = "Gogh"                                # the name of the class that is correlated to the file name identifier
file_name_identifier = "kw"  					# something distinguishable to tell the two images apart.
data_directory = "resized/load/chosen"			# directory to load the train images
load_dir = 'resized/load/checkpoints'           # for the TF graphs


plt_show = True  								# To show the plotted values set to true, to never plot anything set to false
augment = False                                 # Whether or not to augment, if img_size or num_augment has change this must be true



image_size_flat = image_size * image_size  	    # Images are stored in one-dimensional arrays of this length.
image_shape = (image_size, image_size)  		# Tuple with height and width of images used to reshape arrays.
num_classes = 2  								# Number of classes, one class for each of 10 digits.
print_full_np_array = False                     # Make numpy not print full arrays rather than:
												# [0,1,2 ... 97,98,99]
												# [0,1,2 ... 97,98,99]
												# 	    [...]
												# [0,1,2 ... 97,98,99]
												# [0,1,2 ... 97,98,99]


# changes the way to print arrays from numpy
if print_full_np_array == True:
	np.set_printoptions(threshold=np.inf)


def initiate_classify():
	"""
	Function to set all the variables for the classifier to run
	:return: all the variables created.
	"""


	print("Loading data")
	data.test.images, data.test.labels = load_data(data_directory, file_name_identifier, image_size_flat, num_channels, augment=False)
	print("Initiating data")
	data.test = data.test.init()
	data.test._name = "test"
	


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
	# runs the rezise_to_chosen.py in resized/load/ to resize the images to classify
	print("Resizing images")
	# os.system('python3 resized/load/resize_to_chosen.py')
	resize.resize_many(image_size, num_channels, file_name_identifier)
	print("Done")

	# starts the session and initiates the tf global variables
	session = tf.Session()
	session.run(tf.global_variables_initializer())

	# initiates all the variables needed to run the classification
	data, x, x_image, y_true, y_true_cls, layer_conv1, layer_conv2, weights_conv1, weights_conv2, layer_flat, \
	num_features, layer_fc1, layer_fc1, layer_fc2, y_pred, y_pred_cls, cost, optimizer, correct_prediction, \
	accuracy, saver = initiate_classify()

	#sets the name fo the file to read
	load_path = os.path.join(load_dir, 'best_validation')

	# Restores the session in load_dir
	saver.restore(sess=session, save_path=load_path)

	print("Session at {} has been restored successfully".format(load_path))
	


	image1 = data.test.images[0]
	image2 = data.test.images[1]
	# prints weights 
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


if __name__ == "__main__":
	# stuff only to run when not called via 'import' here
	main()
