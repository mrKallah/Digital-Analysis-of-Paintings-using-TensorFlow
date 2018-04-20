from IPython.display import Image, display

Image('images/15_style_transfer_flowchart.png')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image
import os
import cv2

import VGG16
import style_data
import augment as aug



# the layers of the model to load
content_layer_ids = [0]
style_layer_ids = [1]



print_tmp = "####" #this is for testing purposes and can be removed xxx

# disables cpu instruction warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

style_image_path = 'images/style.jpg'
content_image_path = 'images/content.jpg'
num_iterations = 120


def load_image(filename, max_size=None):
	image = PIL.Image.open(filename)

	if max_size is not None:
		# Calculate the appropriate rescale-factor for
		# ensuring a max height and width, while keeping
		# the proportion between them.
		factor = max_size / np.max(image.size)

		# Scale the image's height and width.
		size = np.array(image.size) * factor

		# The size is now floating-point because it was scaled.
		# But PIL requires the size to be integers.
		size = size.astype(int)

		# Resize the image.
		image = image.resize(size, PIL.Image.LANCZOS)

	# Convert to numpy floating-point array.
	return np.float32(image)


def plot_image_big(image):
	# Ensure the pixel-values are between 0 and 255.
	image = np.clip(image, 0.0, 255.0)

	# Convert pixels to bytes.
	image = image.astype(np.uint8)

	# Convert to a PIL-image and display it.
	display(PIL.Image.fromarray(image))


def plot_images(content_image, style_image, mixed_image):
	# Create figure with sub-plots.
	fig, axes = plt.subplots(1, 3, figsize=(10, 10))

	# Adjust vertical spacing.
	fig.subplots_adjust(hspace=0.1, wspace=0.1)

	# Use interpolation to smooth pixels?
	smooth = True

	# Interpolation type.
	if smooth:
		interpolation = 'sinc'
	else:
		interpolation = 'nearest'

	# Plot the content-image.
	# Note that the pixel-values are normalized to
	# the [0.0, 1.0] range by dividing with 255.
	ax = axes.flat[0]
	ax.imshow(content_image / 255.0, interpolation=interpolation)
	ax.set_xlabel("Content")

	# Plot the mixed-image.
	ax = axes.flat[1]
	ax.imshow(mixed_image / 255.0, interpolation=interpolation)
	ax.set_xlabel("Mixed")

	# Plot the style-image
	ax = axes.flat[2]
	ax.imshow(style_image / 255.0, interpolation=interpolation)
	ax.set_xlabel("Style")

	# Remove ticks from all the plots.
	for ax in axes.flat:
		ax.set_xticks([])
		ax.set_yticks([])

	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	plt.show()


def mean_squared_error(a, b):
	return tf.reduce_mean(tf.square(a - b))


def create_content_loss(session, model, content_image, layer_ids):
	"""
	Create the loss-function for the content-image.

	Parameters:
	session: An open TensorFlow session for running the model's graph.
	model: The model, e.g. an instance of the VGG16-class.
	content_image: Numpy float array with the content-image.
	layer_ids: List of integer id's for the layers to use in the model.
	"""

	# Create a feed-dict with the content-image.


	feed_dict = model.create_feed_dict(model, content_image, layer_ids)

	# Get references to the tensors for the given layers.
	layers = model.get_layer_tensors(model, layer_ids)

	# Calculate the output values of those layers when
	# feeding the content-image to the model.



	print("#################### create_content_loss() ####################")

	print("layer_ids = {}".format(layer_ids))
	print("layers = {}".format(layers))

	print("feed_dict = {}".format(feed_dict))
	print("feed_dict['content:0'].shape = {}".format(feed_dict['content:0'].shape))

	print("content_image.shape = {}".format(content_image.shape))
	print("content_image = {}".format(content_image))



	print("###############################################################")
	print(aug.iterate())



	values = session.run(layers, feed_dict=feed_dict)
	print(aug.iterate())

	# Set the model's graph as the default so we can add
	# computational nodes to it. It is not always clear
	# when this is necessary in TensorFlow, but if you
	# want to re-use this code then it may be necessary.
	with model.graph.as_default():
		# Initialize an empty list of loss-functions.
		layer_losses = []

		# For each layer and its corresponding values
		# for the content-image.
		for value, layer in zip(values, layers):
			# These are the values that are calculated
			# for this layer in the model when inputting
			# the content-image. Wrap it to ensure it
			# is a const - although this may be done
			# automatically by TensorFlow.
			value_const = tf.constant(value)

			# The loss-function for this layer is the
			# Mean Squared Error between the layer-values
			# when inputting the content- and mixed-images.
			# Note that the mixed-image is not calculated
			# yet, we are merely creating the operations
			# for calculating the MSE between those two.
			loss = mean_squared_error(layer, value_const)

			# Add the loss-function for this layer to the
			# list of loss-functions.
			layer_losses.append(loss)

		# The combined loss for all layers is just the average.
		# The loss-functions could be weighted differently for
		# each layer. You can try it and see what happens.
		total_loss = tf.reduce_mean(layer_losses)

	return total_loss


def gram_matrix(tensor):
	shape = tensor.get_shape()

	# Get the number of feature channels for the input tensor,
	# which is assumed to be from a convolutional layer with 4-dim.
	num_channels = int(shape[1])

	# Reshape the tensor so it is a 2-dim matrix. This essentially
	# flattens the contents of each feature-channel.
	matrix = tf.reshape(tensor, shape=[-1, num_channels])

	# Calculate the Gram-matrix as the matrix-product of
	# the 2-dim matrix with itself. This calculates the
	# dot-products of all combinations of the feature-channels.
	gram = tf.matmul(tf.transpose(matrix), matrix)

	return gram


def create_style_loss(session, model, style_image, layer_ids):
	"""
	Create the loss-function for the style-image.

	Parameters:
	session: An open TensorFlow session for running the model's graph.
	model: The model, e.g. an instance of the VGG16-class.
	style_image: Numpy float array with the style-image.
	layer_ids: List of integer id's for the layers to use in the model.
	"""
	print("#################### create_style_loss() ####################")

	print("content_image.shape = {}".format(style_image.shape))
	print("content_image = {}".format(style_image))
	print("layer_ids = {}".format(layer_ids))



	# Create a feed-dict with the style-image.
	feed_dict = model.create_feed_dict(model, style_image, layer_ids)

	# Get references to the tensors for the given layers.
	layers = model.get_layer_tensors(model, layer_ids)

	print("layers = {}".format(layers))

	# Set the model's graph as the default so we can add
	# computational nodes to it. It is not always clear
	# when this is necessary in TensorFlow, but if you
	# want to re-use this code then it may be necessary.
	with model.graph.as_default():
		# Construct the TensorFlow-operations for calculating
		# the Gram-matrices for each of the layers.
		gram_layers = [gram_matrix(layer) for layer in layers]



		print("gram_layers = {}".format(gram_layers))
		print("layers = {}".format(layers))
		# print("feed_dict['y_true:0'].shape = {}".format(feed_dict['y_true:0'].shape))

		print("###############################################################")


		# Calculate the values of those Gram-matrices when
		# feeding the style-image to the model.
		values = session.run(gram_layers, feed_dict=feed_dict)

		# Initialize an empty list of loss-functions.
		layer_losses = []
		# For each Gram-matrix layer and its corresponding values.
		for value, gram_layer in zip(values, gram_layers):
			# These are the Gram-matrix values that are calculated
			# for this layer in the model when inputting the
			# style-image. Wrap it to ensure it is a const,
			# although this may be done automatically by TensorFlow.
			value_const = tf.constant(value)
			# The loss-function for this layer is the
			# Mean Squared Error between the Gram-matrix values
			# for the content- and mixed-images.
			# Note that the mixed-image is not calculated
			# yet, we are merely creating the operations
			# for calculating the MSE between those two.
			loss = mean_squared_error(gram_layer, value_const)

			# Add the loss-function for this layer to the
			# list of loss-functions.
			layer_losses.append(loss)


		# The combined loss for all layers is just the average.
		# The loss-functions could be weighted differently for
		# each layer. You can try it and see what happens.
		total_loss = tf.reduce_mean(layer_losses)


	return total_loss


def create_denoise_loss(model):
	out = model.input

	l = out.get_shape()[0]
	a = out[0:l - 1]
	b = out[1:l]
	c = tf.where(a < b, tf.ones_like(a), tf.zeros_like(a))



	print("#################### create_denoise_loss() ####################")
	print("a = {}".format(a))
	print("b = {}".format(b))
	print("c = {}".format(c))
	print("tf.reduce_sum(c) = {}".format(tf.reduce_mean(c)))
	print("###############################################################")





	return tf.reduce_sum(c)



def get_image(path):
	image = cv2.imread(path, 0)
	image = np.asarray(image, dtype="float32")

	image = cv2.resize(image, (100, 100))

	content_image = np.reshape(image, (-1, 10000))

	return content_image


first_run = True
def style_transfer(content_image, style_image, content_layer_ids, style_layer_ids, weight_content=1.5,
					weight_style=10.0, weight_denoise=0.3, num_iterations=120, step_size=10.0):
	"""
	Use gradient descent to find an image that minimizes the
	loss-functions of the content-layers and style-layers. This
	should result in a mixed-image that resembles the contours
	of the content-image, and resembles the colours and textures
	of the style-image.

	Parameters:
	content_image: Numpy 3-dim float-array with the content-image.
	style_image: Numpy 3-dim float-array with the style-image.
	content_layer_ids: List of integers identifying the content-layers.
	style_layer_ids: List of integers identifying the style-layers.
	weight_content: Weight for the content-loss-function.
	weight_style: Weight for the style-loss-function.
	weight_denoise: Weight for the denoising-loss-function.
	num_iterations: Number of optimization iterations to perform.
	step_size: Step-size for the gradient in each iteration.
	"""

	# Create an instance of the VGG16-model. This is done
	# in each call of this function, because we will add
	# operations to the graph so it can grow very large
	# and run out of RAM if we keep using the same instance.

	print("frog")
	global first_run
	if first_run == True:
		model = style_data.init()
	else:
		model = style_data
	first_run = False
	print("bear")




	# Create a TensorFlow-session.
	session = tf.InteractiveSession(graph=model.graph)

	print("#################### style_transfer() ####################")
	# Print the names of the content-layers.
	print("Content layers:")
	print(model.get_layer_names(model, content_layer_ids))
	print()

	# Print the names of the style-layers.
	print("Style layers:")
	print(model.get_layer_names(model, style_layer_ids))
	print()
	print("###############################################################")

	# Create the loss-function for the content-layers and -image.
	loss_content = create_content_loss(session=session,
	                                   model=model,
	                                   content_image=get_image(content_image_path),
	                                   layer_ids=content_layer_ids)

	# Create the loss-function for the style-layers and -image.
	loss_style = create_style_loss(session=session,
	                               model=model,
	                               style_image=get_image(style_image_path),
	                               layer_ids=style_layer_ids)

	# Create the loss-function for the denoising of the mixed-image.
	loss_denoise = create_denoise_loss(model)


	print("#################### style_transfer2() ####################")
	print("loss_denoise = {}".format(loss_denoise))

	# Create TensorFlow variables for adjusting the values of
	# the loss-functions. This is explained below.
	adj_content = tf.Variable(1e-10, name='adj_content')
	adj_style = tf.Variable(1e-10, name='adj_style')
	adj_denoise = tf.Variable(1e-10, name='adj_denoise')

	# Initialize the adjustment values for the loss-functions.
	session.run([adj_content.initializer,
	             adj_style.initializer,
	             adj_denoise.initializer])

	# Create TensorFlow operations for updating the adjustment values.
	# These are basically just the reciprocal values of the
	# loss-functions, with a small value 1e-10 added to avoid the
	# possibility of division by zero.
	update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
	update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
	update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

	# This is the weighted loss-function that we will minimize
	# below in order to generate the mixed-image.
	# Because we multiply the loss-values with their reciprocal
	# adjustment values, we can use relative weights for the
	# loss-functions that are easier to select, as they are
	# independent of the exact choice of style- and content-layers.
	loss_combined = weight_content * adj_content * loss_content + weight_style * adj_style * loss_style + weight_denoise * adj_denoise * loss_denoise

	print("weight_content = {}".format(weight_content))
	print("adj_content = {}".format(adj_content))
	print("loss_content = {}".format(loss_content))
	print("weight_style = {}".format(weight_style))
	print("adj_style = {}".format(adj_style))
	print("loss_style = {}".format(loss_style))
	print("weight_denoise = {}".format(weight_denoise))
	print("adj_denoise = {}".format(adj_denoise))
	print("loss_denoise = {}".format(loss_denoise))
	print("loss_combined = {}".format(loss_combined))


	# Use TensorFlow to get the mathematical function for the
	# gradient of the combined loss-function with regard to
	# the input image.
	gradient = tf.gradients(tf.reduce_mean(loss_combined) + model.input, [loss_combined, model.input], stop_gradients=[loss_combined, model.input])
	# gradient = tf.gradients(loss_combined, model.input)



	# List of tensors that we will run in each optimization iteration.
	run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

	# The mixed-image is initialized with random noise.
	# It is the same size as the content-image.
	mixed_image = np.random.rand(100,100, 3) + 128

	print(mixed_image.shape)



	mixed_image = np.asarray(mixed_image, dtype="float32")
	mixed_image = cv2.cvtColor(mixed_image, cv2.COLOR_RGB2GRAY)
	mixed_image = cv2.resize(mixed_image, (100, 100))
	mixed_image = mixed_image.reshape(-1, 10000)
	print("mixed_image.shape = {}".format(mixed_image.shape))

	for i in range(num_iterations):
		# Create a feed-dict with the mixed-image.
		# feed_dict = model.create_feed_dict3(model, mixed_image)
		# feed_dict = {model.get_tensor_by_name(model, 'style:0'), 'test2:0', mixed_image}


		print("#################### !!!!!!!!!!!!!!!!!! ####################")
		print("mixed_image.shape = {}".format(mixed_image.shape))
		print("content_image.shape = {}".format(content_image.shape))


		print("#################### style_transfer() in loop ####################")
		print("model.input = {}".format(model.input))
		print("model.get_layer_tensors(model, [0]) = {}".format(model.get_layer_tensors(model, [0])))
		print("gradient = {}".format(gradient))
		print("run_list = {}".format(run_list))
		print("mixed_image.shape = {}".format(mixed_image.shape))
		print("content_image.shape = {}".format(content_image.shape))
		print("num_iterations = {}".format(num_iterations))
		print("model = {}".format(model))
		print("mixed_image.shape = {}".format(mixed_image.shape))
		print("run_list = {}".format(run_list))
		print("###############################################################")
		print("layer_id = {}".format([3]))
		print("run_list = {}".format(run_list))

		# print("feed_dict['test2:0'].shape = {}".format(feed_dict['test2:0'].shape))
		# print("feed_dict['test2:0'] = {}".format(feed_dict['test2:0']))
		print("###############################################################")
		# print(tf.get_default_graph().as_graph_def())

		# Use TensorFlow to calculate the value of the
		# gradient, as well as updating the adjustment values.

		mix = "test2:0"
		first = 'content:0'
		secound = 'style:0'

		feed_dict = {first: content_image, secound: style_image, mix: mixed_image}

		run_list = run_list[0][1]
		print(run_list)
		print("style_image.shape = {}".format(style_image.shape))
		print("run_list = {}".format(run_list))
		print("feed_dict = {}".format(feed_dict))

		grad, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list, feed_dict=feed_dict)
		# grad, adj_content_val, adj_style_val, adj_denoise_val = feed_dict={y_previous: y_prev, h_prime_previous: h_prime_prev, "Placeholder:0": np.zeros((100, 1)).as_type(np.float32)}

		# Reduce the dimensionality of the gradient.
		grad = np.squeeze(grad)

		# Scale the step-size according to the gradient-values.
		step_size_scaled = step_size / (np.std(grad) + 1e-8)

		# Update the image by following the gradient.
		mixed_image -= grad * step_size_scaled

		# Ensure the image has valid pixel-values between 0 and 255.
		mixed_image = np.clip(mixed_image, 0.0, 255.0)

		# Print a little progress-indicator.
		print(". ", end="")

		# Display status once every 10 iterations, and the last.
		if (i % 10 == 0) or (i == num_iterations - 1):
			print()
			print("Iteration:", i)

			# Print adjustment weights for loss-functions.
			msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
			print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

			# Plot the content-, style- and mixed-images.
			plot_images(content_image=content_image,
			            style_image=style_image,
			            mixed_image=mixed_image)

	# Close the TensorFlow session to release its resources.
	session.close()

	# Return the mixed-image.
	return mixed_image


def main():
	content_image = load_image(content_image_path, max_size=None)
	style_image = load_image(style_image_path, max_size=300)

	content_image = get_image(content_image_path)
	style_image = get_image(style_image_path)



	img = style_transfer(content_image=content_image,
	                     style_image=style_image,
	                     content_layer_ids=content_layer_ids,
	                     style_layer_ids=style_layer_ids,
	                     weight_content=1.5,
	                     weight_style=10.0,
	                     weight_denoise=0.3,
	                     num_iterations=num_iterations,
	                     step_size=10.0)

	print("\nFinal image:")
	plot_image_big(img)


if __name__ == "__main__":
	# stuff only to run when not called via 'import' here
	main()
