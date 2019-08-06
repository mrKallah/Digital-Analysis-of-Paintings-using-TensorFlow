import cv2
import numpy as np
import random as rand
import os
import glob
import shutil
import random
import math


def save(img, dir_path, name, print_interval=100):
	"""
	Saves the intput image in to a file
	:param img: the image to save
	:param dir_path: Where to save the image
	:param name: The name to save the image with an extension, EG. cat.png
	:param print_interval: how often to print. For mass printing. -1 for no printing
	"""
	# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	if print_interval != -1:
		if _i % print_interval == 0:
			print("Handled {0}/{2} images, currently in path {1}".format(_i, dir_path, get_total_files()))
	cv2.imwrite(os.path.join(dir_path, name), img)


total_files = 0


def set_total_files(new_size):
	"""
	Sets the total amount of files to change. Used to count down how many images are left to handle
	:param new_size: the total amount of images
	"""
	global total_files
	total_files = new_size


def get_total_files():
	"""
	:return: Returns the total amount of images to handle
	"""
	global total_files
	return total_files


def prepare_data(data_directory, image_shape, num_channels, num_augment, file_format=".this_is_the_file_format", sub_dir="augmented/"):
	"""
	This function reads the data from a directory and then augments it into a subdirectory.
	It will name all files exept the ones containing the file name identifier as rand.
	It will augmet the data to num_augment amount of images.
	:param data_directory: the directory to find the files
	:param image_shape: the shape of the out put images without the color chanel
	:param num_channels: the color chanels, 1 for grey, 3 for color
	:param num_augment: this can be 1, 10, 20, 30 and 40 and will make the original image into 1-num_augment images
	:param file_format: a file format of your choise, jpeg, png and jpg are defaults
	:param sub_dir: the subdirectory to put the files in. Default: augmented
	"""

	# read all the png, jpg and jpeg and selected files in the folder
	files = glob.glob(os.path.join(data_directory, "*.{}".format(file_format)))
	files2 = glob.glob(os.path.join(data_directory, "*.{}".format("jpg")))
	files3 = glob.glob(os.path.join(data_directory, "*.{}".format("jpeg")))
	files4 = glob.glob(os.path.join(data_directory, "*.{}".format("png")))
	files5 = []
	if file_format != "png":
		files5 = glob.glob(os.path.join(data_directory, "*.{}".format(file_format)))

	# add all the files into first array
	for f in files2:
		files.append(f)
	for f in files3:
		files.append(f)
	for f in files4:
		files.append(f)
	for f in files5:
		files.append(f)

	# delete and re-create the output path to empty it
	out_path = os.path.join(data_directory, sub_dir)
	if os.path.exists(out_path):
		shutil.rmtree(out_path)

	if not os.path.exists(out_path):
		os.makedirs(out_path)
	
	set_total_files(len(files)*num_augment)
	
	# Go thorough all the files in the files array
	iter = 0
	for f in files:
		# print(f)
		# read the image as color or grey images
		if num_channels == 1:
			image = cv2.imread(f, 0)
		else:
			image = cv2.imread(f, num_channels)

		# convert the image to a numpy array of float32s
		image = np.asarray(image, dtype="float32")

		# set the name to be the identifier as default
		#name = file_name_identifier
		name = f.replace("resized/train/", "")
		name = name.replace("resized/validate/", "")
		name = name.replace("resized/test/", "")
		name = name.replace(".png", "")
		name = name.replace(".jpg", "")
		name = name.replace(".jpeg", "")
		name = name.replace(file_format, "")

		# change the name identifier based on if it is in the name of tte file or not
		#f = files[iter]
		#if f.find(file_name_identifier) == -1:
			#name = "rand"

		# resize the image to the propper shape
		image = resize(image, image_shape)

		# augment the images to more images
		# augment_switch(image, out_path, name, num_channels, num_augment)

		augment_many(np.copy(image), num_augment, out_path, name)

		# increase the current iteration
		iter = iter + 1


# first time is -1+1 = 0
_i = -1


def iterate(change=0, to_change=False):
	"""
	This iterates either the global i variable within the class or if to_change is True then it iterates the change
	:param change: the value to iterate if to_change is True
	:param to_change: Whether or not to iterate the global i or the passed in change variable.
	:return: returns i+1 if to_change is False and change+1 if to_change is True
	"""
	global _i
	if to_change:
		change += 1
		return change
	else:
		_i += 1
		return _i


def re_iterate():
	"""
	Restarts the iteration. Sets i back to its default value
	"""
	global _i
	_i = -1


def resize(img, shape):
	"""
	resizes an image to the shape given. Ignores color channels. (50, 50) works for resizing both a color image and
	greyscale image to 50 by 50 and keeps the original color channels
	:param img: the image to resize
	:param shape: the new shape as a tuple
	:return: the resized image
	"""
	cpy = np.copy(img)
	cpy = cv2.resize(cpy, shape)
	return cpy


def crop(img, left, right, top, bottom, new_shape=[False]):
	"""
	crops the images sides and reshapes the
	:param img: the image to crop
	:param left: how many pixels to crop off the left
	:param right: how many pixels to crop off the right
	:param top: how many pixels to crop off the top
	:param bottom: how many pixels to crop off the bottom
	:param new_shape: the new shape of the image, without this the original shape will be retained
	:return: the modified image
	"""
	# copies the image
	cpy = np.copy(img)

	# checks if the it should use the original shape or a new shape
	if new_shape==[False]:
		new_row = cpy.shape[0]
		new_col = cpy.shape[1]
	else:
		new_row = new_shape[0]
		new_col = new_shape[1]

	# gets the rows and cols to use for cropping
	rows = cpy.shape[1]
	cols = cpy.shape[0]

	# crops the image
	cropped = cpy[right:rows-left, top:cols-bottom]

	# resizes the image
	cpy = resize(cropped, (new_row, new_col))

	return cpy


def rotate(img, degree):
	"""
	rotates the image by x degrees
	:param img: the image to rotate
	:param degree: amount of degrees to rotate
	:return: the modified image
	"""
	# copies the image
	cpy = np.copy(img)

	# saves the rows and cols
	rows = cpy.shape[1]
	cols = cpy.shape[0]

	# generates a rotation matrix
	rotation = cv2.getRotationMatrix2D((cols/2,rows/2), degree, 1)

	# rotate the image
	cpy = cv2.warpAffine(cpy, rotation, (cols, rows))

	return cpy


def mirror(img):
	"""
	this mirrors the image
	:param img: the image to mirror
	:return: the mirrored image
	"""
	# creates a copy of the image
	cpy = np.copy(img)

	# mirrors and then returns the image
	return cv2.flip(cpy, 1)


def add_salt_and_pepper_noise(img, _range):
	"""
	adds salt and pepper noise to the image. This is not true to a normal salt and pepper as it adds noise
	in a range from 0 to 255 rather then either 0 or 255
	:param img: the image to add noise to
	:param _range: how often to add noise. Lower means more noise. 0 means every pixel becomes noise and
					100 means every 100th pixel becomes noise
	:return: returns the augmented image
	"""

	# creates a copy of the image
	cpy = np.copy(img)

	# saves the shape in cols and rows
	cols=cpy.shape[0]
	rows = cpy.shape[1]

	# initiates iteration variables
	i = 0
	j = 0

	# iterates through all the pixels in the image and changes its value if a random check gets met
	while i < rows:
		while j < cols:
			if rand.randrange(0, _range) == 0:
				cpy[i][j] = rand.randrange(0,255)
			j = j + 1
		j=0
		i = i + 1

	return cpy


def add_tutti_frutty_noise(img, _range):
	"""
	Adds a similar noise to the salt and pepper noise however rather than being greyscale it is in color.
	:param img: the image to add noise to
	:param _range: how often to add noise. Lower means more noise. 0 means every pixel becomes noise and
					100 means every 100th pixel becomes noise
	:return: the augmented image
	"""

	# creates a copy of the image
	cpy = np.copy(img)

	# saves the shape in cols and rows
	w = cpy.shape[1]
	h = cpy.shape[0]

	# initiates iteration variables
	i = 0
	j = 0

	# iterates through all the pixels in the image and changes its value if a random check gets met
	while i < w:
		while j < h:
			if rand.randrange(0, _range) == 0:
				cpy[i][j][rand.randrange(0,2)] = rand.randrange(0,255)
			j = j + 1
		j=0
		i = i + 1

	return cpy


def sharpen(img, hardness=1.):
	"""
	shapens the image
	:param img: the image to sharpen
	:param hardness: how much the image is sharpened
	:return: returns the shaprened image
	"""

	#creates a copy of the image
	cpy = np.copy(img)

	# creates the shapening kernel
	side = hardness
	centre = (side * 4) + 1
	corner = 0

	kernel = np.array([
		[corner, -side, corner],
		[-side, centre,  -side],
		[corner, -side, corner]])


	# applies the kernel to the image
	cpy = cv2.filter2D(cpy, -1, kernel)

	return cpy


def blur(img, blur_rate=5.0):
	"""
	adds a gaussian blur to the image
	:param img: the image to blur
	:return: the blurred image
	"""
	# Copies the image
	cpy = np.copy(img)


	kernel = np.ones((blur_rate, blur_rate), np.float32) / (blur_rate * blur_rate)

	# applies the kernel
	cpy = cv2.filter2D(cpy, -1, kernel)

	return cpy


def brighten_darken(img, mult):
	"""
	Brightens or darkens the image based on the multiplier
	:param img: the image to augment
	:param mult: abve 1 to brighten, below 1 to darken
	:return: returns the augmented image
	"""
	# copies the image
	cpy = np.copy(img)

	# creates the kernel
	kernel = np.array([mult])

	# applies the kernel to the image
	cpy = cv2.filter2D(cpy, -1, kernel)


	return cpy


def add_edge(img, edge, num_channels, use_alpha=False):
	"""
	adds an edge around the image, used for the affine transform
	:param img: the image to add an edge to
	:param edge: the size of the edge you want
	:param num_channels: number of channels
	:param use_alpha: Whether or not to use alpha for the image. Returns 4 channels if use_alpha is true
	:return: original image with an edge
	"""

	# creates a copy of the image
	cpy = np.copy(img)

	# saves the rows and colums
	rows = cpy.shape[0]
	cols = cpy.shape[1]

	if use_alpha == True:
		num_channels = 4

	# creates the new image size dependant on if it is a color image or not
	if num_channels == 1:
		base_size = cols + (edge * 2), rows + (edge * 2)
	else:
		base_size = cols + (edge * 2), rows + (edge * 2), num_channels

	# make an image for base which is larger than the original img
	if use_alpha == False:
		base = np.zeros(base_size, dtype=np.uint8)
	else:
		base = np.zeros(base_size, dtype=np.uint8)
		if num_channels == 1:
			cpy = cv2.cvtColor(cpy, cv2.COLOR_GRAY2RGBA)
		else:
			cpy = cv2.cvtColor(cpy, cv2.COLOR_RGB2RGBA)
			cpy[:, :, 3] = np.ones(cpy[:, :, 3].shape)
			cpy[:, :, 3] = cpy[:, :, 3] * 255


	# adds the original image to the base image
	base[edge:edge + rows, edge:edge + cols] = cpy

	return base


def affine_transform(img, original_points=np.float32([[0, 0], [0, 500], [500, 500]]), new_points=np.float32([[0, 0], [0, 500], [500, 501]])):
	"""
	Does the a affine transform on the input image. This means parallel lines stay parallel
	:param img: image to transform
	:param original_points: a set of 3 points to be used as reference
	:param new_points: a set of tree new points that the original points gets moved to
	:return: transformed image
	"""
	# creates the rotation matrix
	rotation = cv2.getAffineTransform(original_points, new_points)

	# applies the rotation matrix and transforms the image
	# the rows and cols must be used as shape = (x, x, x) for color images and func is expecting (x, x)
	rows = img.shape[1]
	cols = img.shape[0]
	dst = cv2.warpAffine(img,rotation,(cols,rows))

	return dst


def blend_transparent(base_img, overlay_img):
	"""
	blends two transparent images in accordance to the alpha channel.
	:param base_img: the base image to be overlaid
	:param overlay_img: the image to overlay the base image
	:return: one image containing content from both the imput images
	"""

	# Split out the transparency mask from the colour info
	overlay_img = overlay_img[:, :, :3] # Grab the BRG planes
	overlay_mask = overlay_img[:, :, 3:]  # And the alpha plane

	# Again calculate the inverse mask
	background_mask = 255 - overlay_mask

	# Turn the masks into three channel, so we can use them as weights
	overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
	background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

	# Create a masked out face image, and masked out overlay
	# We convert the images to floating point in range 0.0 - 1.0
	face_part = (base_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
	overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

	# And finally just add them together, and rescale it back to an 8bit integer image
	return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def perspective_transform(img, original_points=np.float32([[0, 0], [0, 500], [500, 0], [500, 500]]), new_points=np.float32([[0, 0], [0, 500], [500, 0], [500, 501]])):
	"""
	Does a perspective transformation on the input image
	:param img: the image to get transformed
	:param original_points: a set of 4 points to be used as a baseline for the transformation
	:param new_points: a set of 4 new points that signifies how much to move the original_points.
	:return: a transformed image
	"""

	# creates the rotation matrix
	rotation = cv2.getPerspectiveTransform(original_points,new_points)

	# applies the rotation matrix and transforms the image
	# the rows and cols must be used as shape = (x, x, x) for color images and func is expecting (x, x)
	rows = img.shape[1]
	cols = img.shape[0]
	dst = cv2.warpPerspective(img, rotation, (cols, rows))
	return dst


def read_image(img_size=300):
	"""
	reads an image from file
	:param img_size: the size to have the image.
	:return: image as nparray with new dimensions
	"""
	img = cv2.imread("test_data/org_large.jpg", 3)
	img = cv2.resize(img, (img_size, img_size))
	return img


def plot_image(img, channels=3, title="Test"):
	"""
	Plots an image to a matplotlib window
	:param img: the image to plot
	:param channels: how many channels the image has
	:param title: the title of the image when plotted
	"""
	import Utils as utils
	utils.plot_image(img, (img.shape[0], img.shape[1],), True, channels, title)


def get_unique_list(return_amount, size_of_list):
	"""
	Creates a unique list in an array.

	example:
	get_unique_list(2, 5) = [array([0, 0, 0, 0, 1]), array([0, 0, 0, 1, 0])]
	get_unique_list(5, 2) = [array([0, 1]), array([1, 0]), array([1, 1]), array([1, 2]), array([2, 1])]

	:param return_amount: the number of arrays to return
	:param size_of_list: the amount of values within the array to return
	:return: returns a list that can be used as parameters for a function
	"""

	if return_amount <= 0:
		exit("return_amount cannot be negative or zero")
	size = []

	# creates an array of 0 to 10, used for the binary counter
	for j in range(10):
		size.append(j)

	import itertools
	indices = []
	# creates a binary like counter but for the 10 digit system, list of (16, 4)
	for j in itertools.product([0, 1], repeat=size_of_list):
		indices.append(j)

	# makes the list larger by adding a changed version of the indices list to itself.
	# this is done by making a copy of the indices to which you add the highest
	# number from the original list, in this case that's the last number added to the indices,
	# to all the values in the copy. Then you append that to the original indices.
	while np.asarray(indices).shape[0] < return_amount:
		cpy = np.copy(indices)

		last = indices[cpy.shape[0] - 1]
		last = np.asarray(last)
		last = last.max()

		cpy = cpy + last
		first_run = True
		for c in cpy:
			if first_run == True:
				first_run = False
			else:
				indices.append(c)

	# moves the data from the indices to the cpy object in order to make the
	# function return indices which is easier to read at a glance
	cpy = np.copy(indices)
	indices = []

	# make the size of the indices array exactly the same as return_amount.
	# this is to make sure the list is not too long
	i = 0
	for f in cpy:
		if i != 0:
			indices.append(f)
		i += 1
		if i == return_amount+1:
			return indices
	return  indices


def crop_many(img, return_amount):
	"""
	Crops the image many times using different crop variables each time.
	Returned image will be resized to be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been cropped and resized back to original size
	"""
	indices = get_unique_list(return_amount, 4)
	images = []

	for i in indices:
		images.append(crop(img, i[0], i[1], i[2], i[3]))

	return images


def rotate_many(img, return_amount):
	"""
	Rotates the image many times using different rotation variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been rotated
	"""

	images = []
	unique = get_unique_list(return_amount+2, 1)

	# adds clockwise and counter clockwise rotated images to images array
	i = 0
	for i in range(0, int((return_amount / 2) + 1)):
		images.append(rotate(img, unique[i] / 10))
		images.append(rotate(img, -unique[i] / 10))

	# if the return_amount is an odd number, the last image must be added manually
	i = 0
	while np.asarray(images).shape[0] < return_amount:
		images.append(rotate(img, unique[i+1] / 10))

	cpy = np.copy(images)
	images = []
	for i in range(0, return_amount):
		images.append(cpy[i])


	images = np.asarray(images)
	return images


def affine_many(img, return_amount):
	"""
	Affine transforming the image many times using different transformation variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been transformed
	"""

	# gets the list of numbers to use as rotational values for hte affine transformation.
	# converts into float32 and (3, 2) shape to get the minimum 3 points needed for affine transformations.
	indices = get_unique_list(return_amount, 6)
	indices = np.asarray(indices, np.float32)
	indices = indices.reshape(-1, 3, 2)

	# does the affine transform and adds it to array.
	images = []
	for i in indices:
		# The original 3 points are created, the indices are then added to this point, to make the transformation
		# very small. A mix of large and small numbers are good for org_point as that makes for less drastic change
		# per increased number per point.
		org_points = np.float32([[0, 0], [0, 500], [500, 500]])
		new_points = org_points + i

		affine = affine_transform(img, original_points=org_points, new_points=new_points)
		images.append(affine)

	return images


def perspective_many(img, return_amount):
	"""
	Perspective transforming the image many times using different transformation variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been transformed
	"""

	# gets the list of numbers to use as rotational values for hte affine transformation.
	# converts into float32 and (3, 2) shape to get the minimum 3 points needed for affine transformations.
	indices = get_unique_list(return_amount, 8)
	indices = np.asarray(indices, np.float32)
	indices = indices.reshape(-1, 4, 2)

	# does the affine transform and adds it to array.
	images = []
	for i in indices:
		# The original 3 points are created, the indicies are then added to this point, to make the transformation
		# very small. A mix of large and small numbers are good for org_point as that makes for less drastic change
		# per increased number per point.
		org_points = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])
		new_points = org_points + i

		affine = perspective_transform(img, original_points=org_points, new_points=new_points)
		images.append(affine)

	return images


def brighten_darken_many(img, return_amount):
	"""
	Brighten or darken the image many times using different function variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been brightened or darkened
	"""
	numbers = []
	for i in range(1, int((return_amount / 2) + 2)):
		numbers.append(((i * 3) / 1000) + 1)
		numbers.append(1 - ((i * 3) / 1000))

	images = []
	for i in range(0, return_amount):
		images.append(brighten_darken(img, numbers[i]))

	return images


def blur_many(img, return_amount):
	"""
	Blur the image many times using different blur variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been blurred
	"""
	images = []
	for i in range(2, return_amount+2):
		images.append(blur(img, i))

	return images


def sharpen_many(img, return_amount):
	"""
	Sharpen the image many times using different sharpen variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been sharpened
	"""
	images = []
	for i in range(1, return_amount+1):
		images.append(sharpen(img, 0 + (i/10)))

	return images


def add_salt_and_pepper_noise_many(img, return_amount):
	"""
	add salt and pepper to the image many times using different function variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been added salt and pepper to
	"""
	images = []
	for i in range(0, return_amount):
		images.append(add_salt_and_pepper_noise(img, 1000-i))
	return images


def add_tutti_frutty_noise_many(img, return_amount):
	"""
	adds tutto frutty noise to the image many times using different function variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:return: returns an array of images that have been changed
	"""
	images = []
	for i in range(0, return_amount):
		images.append(add_tutti_frutty_noise(img, 1000-i))
	return images


def piecewise_perspective_transform_many(img, return_amount, random_range=-1):
	"""
	piecewise perspective transform the image many times using different transformation variables each time.
	Returned image will be same size as original
	:param img: the image to change
	:param return_amount: the amount of images to return
	:param random_range: the size of the transformations.
			-1 by default which equates to the average of the height and width of the image / 15
	:return: returns an array of images that have been transformed
	"""
	if random_range == -1:
		random_range = int(((img.shape[0] + img.shape[1]) / 2)/ 15)
	elif random_range < -1:
		exit("Error in piecewise_perspective_transform_many(). Random range cannot be negative value exept -1")

	images = []
	for i in range(0, return_amount):
		images.append(piecewise_perspective_transform(img, random_range))
	return images


def two_to_four_points(two_points):
	"""
	takes an array of two arrays with 2 values and makes it into a array of 8 values
	example:
	two_points = [[0, 1], [2, 3]]
	four_points = [0, 1, 0, 3, 2, 1, 2, 3]

	:param two_points: the points to get converted
	:return: returns an array of 8 values
	"""
	four_points = two_points[0][0], two_points[0][1], two_points[0][0], two_points[1][1], two_points[1][0], \
					two_points[0][1], two_points[1][0], two_points[1][1]
	return np.asarray(four_points)


def get_nine_segments_from_one_image(img):
	"""
	divides an image into 9 equal parts and returns it as an array
	:param img: the image to cut into 9 parts
	:return: an array of the 9 segments of the image
	"""

	# divide the images size into 3 to have 3 points on the image
	quadrants = img.shape[0] / 3
	quadrants = int(quadrants)

	# original image gets segment into smaller pieces, of 1/3 the size.
	tmp = []
	total = 0
	for j in range(0, 3):
		for k in range(0, 3):
			cropped = img[quadrants * j:quadrants * (j + 1), quadrants * k:quadrants * (k + 1)]
			tmp.append(cropped)
			total += 1

	images = np.asarray(tmp)

	return images


def get_two_sets_of_4x2_coordinates(image_size, random_range=10):
	"""
	Creates coordinates for perspective transforming images
	:param image_size: the size of the images to transform
	:param random_range: the maximum amount to move the points by
	:return: old and new points for perspective transforming an image that has been divided into 9 segments.
	"""
	# creates a list of (?,2) coordinates that equates to two of the opposite corners of all the segments
	tmp_coords = []
	for i in range(0, 3):
		for j in range(0, 3):
			# tmp_coords.append([[j, i], [j + 1, i + 1]])
			tmp_coords.append([[0, 0], [1, 1]])
	original_coordinates = np.asarray(tmp_coords) * (image_size / 3)

	# converts the line to a square by calculating the correct points using the 2 known points
	# then it converts the coordinates from shape (8,1) to (4,2) for compatibility with perspective transform
	tmp_coords = []
	for c in original_coordinates:
		tmp_coords.append(two_to_four_points(c))
	original_coordinates = np.asarray(tmp_coords)
	original_coordinates = original_coordinates.reshape((9, 4, 2))

	# crates an array for how much the middle of the image should move.
	move = []
	for i in range(0, 4):
		if random.randrange(0, 1) == 0:
			move.append([[random.randrange(0-random_range, random_range), random.randrange(0-random_range, random_range)]])
	move = np.asarray(move)

	# moves the middle of the image around but leaving the edges flush
	new_coordinates = np.copy(original_coordinates)

	# If you have an image of 300, 300
	# moves the 4 image corners at ca 100, 100
	new_coordinates[0][3][0] = new_coordinates[0][3][0] - move[0][0][1]
	new_coordinates[0][3][1] = new_coordinates[0][3][1] - move[0][0][0]
	new_coordinates[1][1][0] = new_coordinates[1][1][0] - move[0][0][1]
	new_coordinates[1][1][1] = new_coordinates[1][1][1] - move[0][0][0]
	new_coordinates[3][2][0] = new_coordinates[3][2][0] - move[0][0][1]
	new_coordinates[3][2][1] = new_coordinates[3][2][1] - move[0][0][0]
	new_coordinates[4][0][0] = new_coordinates[4][0][0] - move[0][0][1]
	new_coordinates[4][0][1] = new_coordinates[4][0][1] - move[0][0][0]

	# moves the 4 image corners at ca 200, 100
	new_coordinates[1][3][0] = new_coordinates[1][3][0] - move[1][0][1]
	new_coordinates[1][3][1] = new_coordinates[1][3][1] - move[1][0][0]
	new_coordinates[2][1][0] = new_coordinates[2][1][0] - move[1][0][1]
	new_coordinates[2][1][1] = new_coordinates[2][1][1] - move[1][0][0]
	new_coordinates[4][2][0] = new_coordinates[4][2][0] - move[1][0][1]
	new_coordinates[4][2][1] = new_coordinates[4][2][1] - move[1][0][0]
	new_coordinates[5][0][0] = new_coordinates[5][0][0] - move[1][0][1]
	new_coordinates[5][0][1] = new_coordinates[5][0][1] - move[1][0][0]

	# moves the 4 image corners at ca 100, 200
	new_coordinates[3][3][0] = new_coordinates[3][3][0] - move[2][0][1]
	new_coordinates[3][3][1] = new_coordinates[3][3][1] - move[2][0][0]
	new_coordinates[4][1][0] = new_coordinates[4][1][0] - move[2][0][1]
	new_coordinates[4][1][1] = new_coordinates[4][1][1] - move[2][0][0]
	new_coordinates[6][2][0] = new_coordinates[6][2][0] - move[2][0][1]
	new_coordinates[6][2][1] = new_coordinates[6][2][1] - move[2][0][0]
	new_coordinates[7][0][0] = new_coordinates[7][0][0] - move[2][0][1]
	new_coordinates[7][0][1] = new_coordinates[7][0][1] - move[2][0][0]

	# moves the 4 image corners at ca 200, 200
	new_coordinates[4][3][0] = new_coordinates[4][3][0] - move[3][0][1]
	new_coordinates[4][3][1] = new_coordinates[4][3][1] - move[3][0][0]
	new_coordinates[5][1][0] = new_coordinates[5][1][0] - move[3][0][1]
	new_coordinates[5][1][1] = new_coordinates[5][1][1] - move[3][0][0]
	new_coordinates[7][2][0] = new_coordinates[7][2][0] - move[3][0][1]
	new_coordinates[7][2][1] = new_coordinates[7][2][1] - move[3][0][0]
	new_coordinates[8][0][0] = new_coordinates[8][0][0] - move[3][0][1]
	new_coordinates[8][0][1] = new_coordinates[8][0][1] - move[3][0][0]

	new_coordinates = np.asarray(new_coordinates, np.float32)
	original_coordinates = np.asarray(original_coordinates, np.float32)

	return original_coordinates, new_coordinates


def perspective_transform_with_edge(img, random_range, original_points=np.float32([[0, 0], [0, 500], [500, 0], [500, 500]]), new_points=np.float32([[0, 0], [0, 500], [500, 0], [500, 501]])):
	"""
	perspective transforms the image but instead of cutting the edges that go outside the bounds of the image size it
	resizes the image to keep all the content. Used for piecewise perspective transformations

	:param img: the image to transform
	:param random_range: the max amount to change the points by
	:param original_points: an array of 4x2 points that corresponds to coordinates in the image
	:param new_points: an array of 4x2 points that corresponds to the new coordinates the original_points have moved to
	:return: returns the transformed image, NOTE: returned image size != original image size
	"""
	# creates the rotation matrix
	rotation = cv2.getPerspectiveTransform(original_points, new_points)
	if img.shape == (img.shape[0], img.shape[1]):
		channels = 1
		img = add_edge(img, random_range, channels, use_alpha=False)
	else:
		channels = img.shape[2]
		img = add_edge(img, random_range, channels, use_alpha=True)

	# applies the rotation matrix and transforms the image
	# the rows and cols must be used as shape = (x, x, x) for color images and func is expecting (x, x)
	rows = img.shape[1]
	cols = img.shape[0]
	dst = cv2.warpPerspective(img,rotation,(cols,rows))
	return dst


def stitch(images, channels):
	"""
	Stitches two images based on alpha layers
	:param images: an array of the images to be stitched
	:param channels: the number of channels in the images
	:return:
	"""
	x = images[0]
	row = x.shape[0]
	col = x.shape[1]

	if channels == 1:
		final = np.zeros(x.shape)

		masks = []
		for m in images:
			mask = np.copy(m)
			for x in range(0, row):
				for y in range(0, col):
					if mask[x][y] > 1:
						mask[x][y] = 255
					else:
						mask[x][y] = 0
			masks.append(mask)

		plot_image(masks[0])

		for i in range(0, row):
			for j in range(0, col):
				if masks[0][i][j] == 255.0:
					final[i][j] = images[0][i][j]
				elif masks[1][i][j] == 255.0:
					final[i][j] = images[1][i][j]
				elif masks[2][i][j] == 255.0:
					final[i][j] = images[2][i][j]
				elif masks[3][i][j] == 255.0:
					final[i][j] = images[3][i][j]
				elif masks[4][i][j] == 255.0:
					final[i][j] = images[4][i][j]
				elif masks[5][i][j] == 255.0:
					final[i][j] = images[5][i][j]
				elif masks[6][i][j] == 255.0:
					final[i][j] = images[6][i][j]
				elif masks[7][i][j] == 255.0:
					final[i][j] = images[7][i][j]
				elif masks[8][i][j] == 255.0:
					final[i][j] = images[8][i][j]
				else:
					if i > row - 10 & j > col - 10:
						final[i][j] = final[i + 5][j + 5]
					else:
						final[i][j] = final[i - 5][j - 5]
	else:
		final = np.zeros(x.shape)
		for i in range(0, row):
			for j in range(0, col):
				# print(layered[0][i][j][channels])
				if images[0][i][j][channels] == 255.0:
					final[i][j] = images[0][i][j]
				elif images[1][i][j][channels] == 255.0:
					final[i][j] = images[1][i][j]
				elif images[2][i][j][channels] == 255.0:
					final[i][j] = images[2][i][j]
				elif images[3][i][j][channels] == 255.0:
					final[i][j] = images[3][i][j]
				elif images[4][i][j][channels] == 255.0:
					final[i][j] = images[4][i][j]
				elif images[5][i][j][channels] == 255.0:
					final[i][j] = images[5][i][j]
				elif images[6][i][j][channels] == 255.0:
					final[i][j] = images[6][i][j]
				elif images[7][i][j][channels] == 255.0:
					final[i][j] = images[7][i][j]
				elif images[8][i][j][channels] == 255.0:
					final[i][j] = images[8][i][j]
				else:
					if i > row - 10 & j > col - 10:
						final[i][j] = final[i + 1][j + 1]
					else:
						final[i][j] = final[i - 1][j - 1]

	return final


def stitch_nine_images(images, random_range):
	"""
	stiches nine images into one. Used for the piecewise perspective transforms.
	:param images: an array of the image segments
	:param random_range: the max amount to change the points by
	:return: returns one image that is the 9 segments stitched together
	"""
	# sets the number of channels, without crashing. Greyscale images doesnt have a shape[2]
	img = images[0].shape
	if img == (img[0], img[1]):
		channels = 1
	else:
		channels = img[2]
	# above 1 to brighten, below to darken
	if channels == 1:
		blank = np.zeros((images[0].shape[0] * 3, images[0].shape[1] * 3))
	else:
		blank = np.zeros((images[0].shape[0] * 3, images[0].shape[1] * 3, images[0].shape[2]))

	edge = random_range * 2
	cols = images[0].shape[0] - edge
	rows = images[0].shape[1] - edge

	imgs = []
	for img in images:
		imgs.append(cv2.resize(img, (rows + edge + 2, cols + edge + 2)))
	images = np.asarray(imgs)

	k = 0
	layered = []

	# add an edge to the image to make the image the same size as the output image
	for _iter in range(0, 3):
		for j in range(0, 3):
			cpy = np.copy(blank)
			cpy[_iter * cols:((_iter + 1) * cols) + edge, j * rows:((j + 1) * rows) + edge] = \
				images[k][1:images[k].shape[0]-1, 1:images[k].shape[1]-1]
			layered.append(cpy)
			k += 1

	if channels == 1:
		final = stitch(layered, 1)
	else:
		final = stitch(layered, 3)

	return final


def piecewise_perspective_transform(image, random_range=10):
	"""
	Takes an image and divides it into 9 pieces:

	ABC
	DEF
	GHI

	It then takes the corners of E and moves them randomly
	It moves the corners that connect to the corners of E and moves them accordingly
	It then stitches the image back together

	:param image: the image to transform
	:param random_range: the max amount to move the E image corners by in terms of pixels
	:return: returns a transformed image
	"""

	# images must be same sided for this function
	assert image.shape[0] == image.shape[1]
	images = get_nine_segments_from_one_image(image)

	##########################
	#        Add warp        #
	##########################

	original_coordinates, new_coordinates = get_two_sets_of_4x2_coordinates(image.shape[0], random_range)

	# applies the random perspective transformations to the image segments by using the new and original coordinates
	tmp_img = []
	j = 0
	for im in images:
		tmp_img.append(perspective_transform_with_edge(im, random_range, original_coordinates[j], new_coordinates[j]))
		j += 1
	images = np.asarray(tmp_img)

	###########################
	#     Stitch together     #
	###########################

	final = stitch_nine_images(images, random_range)

	final = crop(final, random_range * 5, random_range, random_range, random_range * 5, (image.shape[0], image.shape[0]))

	return final


def augment_many(original_image, return_amount, output_path, name):
	"""
	Augments an input image into return_amount number of images.
	:param original_image: the image to augment
	:param return_amount: the amount of images to return (this can be inaccurate up to 9 images due to rounding)
	:param output_path: the path to put the images in
	:param name: the output name of the images. Must not contain file extension
	:return:
	"""
	images = []

	_return_amount = math.ceil(return_amount / 10)

	images.append(add_tutti_frutty_noise_many(np.copy(original_image), _return_amount))
	images.append(add_salt_and_pepper_noise_many(np.copy(original_image), _return_amount))
	images.append(sharpen_many(np.copy(original_image), _return_amount))
	images.append(blur_many(np.copy(original_image), _return_amount))
	images.append(brighten_darken_many(np.copy(original_image), _return_amount))
	images.append(piecewise_perspective_transform_many(np.copy(original_image), _return_amount))
	images.append(perspective_many(np.copy(original_image), _return_amount))
	images.append(crop_many(np.copy(original_image), _return_amount))
	images.append(affine_many(np.copy(original_image), _return_amount))
	images.append(rotate_many(np.copy(original_image), _return_amount))

	for img in images:
		for im in img:
			save(np.asarray(im), output_path, (name + "{}.png".format(iterate())))

	return images


def main():
	"""
	This is a function used for testing the functionality of the augmentations
	"""
	img = read_image()
	img = cv2.resize(img, (300, 300))

	copy = np.copy(img)

	directory = "tmp_img/augmented/"

	if not os.path.exists(directory):
		os.makedirs(directory)

	copy = augment_many(copy, 1, directory, "kw_74")

	# plot_image(img)
	for cpy in copy:
		for c in cpy:
			plot_image(c)


if __name__ == "__main__":
	# stuff only to run when not called via 'import' here
	main()
