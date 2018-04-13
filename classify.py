
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
	data.test.images, data.test.labels = load_data(data_directory, file_name_identifier, image_size_flat, num_channels, augment=False)
	data.test = data.test.init()
	data.test._name = "test"

	print(data.test.labels)

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


def main():
	# ensure we use the global variuables rather than local variables
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


	load_dir = 'resized/load/checkpoints'
	if not os.path.exists(load_dir):
	    os.makedirs(load_dir)

	load_path = os.path.join(load_dir, 'best_validation')
	saver.restore(sess=session, save_path=load_path)

	print("Session at {} has been restored successfully".format(load_path))

	image = data.test.images

	# prints accuracy after optimization
	utils.print_prediction(data, 1, x, y_true, session, y_pred_cls, class_one, class_zero, image, image_shape, plt_show)


if __name__ == "__main__":
	# stuff only to run when not called via 'import' here
	main()
