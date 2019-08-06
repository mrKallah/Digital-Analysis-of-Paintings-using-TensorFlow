import cv2
import os
import shutil

def get_paths():
	"""
	Gets the folders ready
	:return: a string with the input_path, and all the files in the input_path in an array
	"""
	# Gets the current working directory
	input_path = os.path.dirname(os.path.realpath(__file__))

	# gets the output path
	output_path = os.path.join(input_path, "chosen")

	# removes and recreates the output path to make sure the images in it are removed
	if os.path.exists(output_path):
		shutil.rmtree(output_path)
	os.makedirs(output_path)

	# Gets the files in the directory into an array
	files_in_input_dir = os.listdir(input_path)

	return input_path, files_in_input_dir


def bad_extention(infile):
	"""
	Tests if the files in the infile has extensions that are known as image files
	Used to ignore folders and non-image files
	:param infile: the folder to check images from
	:return: boolean value to say if you should ignore it
	"""
	# checks if the file extensions are known
	filename, file_extension = os.path.splitext(infile)
	need_return = True
	if file_extension == ".png":
		need_return = False
	elif file_extension == ".jpeg":
		need_return = False
	elif file_extension == ".jpg":
		need_return = False
	return need_return

# resizes the input image to X by Y size and make them grayscale
def _resize_and_write(infile, img_size, name, num_chanels, input_path):
	"""
	Resizes an image and writes it to the disk
	:param infile: the file in the input_path folder to read
	:param img_size: the new size of the images
	:param name: name to write files as
	:param num_chanels: amount of color channels: 1 = greyscale, 3 = color
	:param input_path: the folder to get the images from
	:return:
	"""
	if bad_extention(infile):
		return
	

	# reads the input image from the relative path
	image = cv2.imread(os.path.join(input_path, infile), num_chanels)

	# resizes the image, different for color and greyscale
	if num_chanels == 1:
		image = cv2.resize(image, (img_size, img_size))
	else:
		image = cv2.resize(image, (img_size, img_size), 3)

	# writes the image to the relative new path
	cv2.imwrite(os.path.join(input_path, "chosen", "{}.png".format(name)), image)

def rm_ext(label):
	label = label.replace(".png", "")
	label = label.replace(".jpg", "")
	label = label.replace(".jpeg", "")
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


def resize_many(img_size=50, num_chanels=1):

	# goes through all the input images and resizes and saves them
	input_path, files_in_input_dir = get_paths()

	iteration = 0
	# goes through all the images in the input folder and resizes them and saves them to the output folder
	for infile in files_in_input_dir:

		# calls the resize function to resize the file.
		_resize_and_write(infile, img_size, "{0}{1}".format(rm_ext(infile), iteration), num_chanels, input_path)

		iteration = iteration + 1

if __name__ == "__main__":
	# stuff only to run when not called via 'import' here
	resize_many()
