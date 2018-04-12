import cv2
import os
import numpy as np
import shutil


# size of output images
x, y = 100,100
use_file_name = False
name_convension = "rand" #Folder name of input images, can only be one class


# Gets the current working directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Gets the folder the images are in
input_path = dir_path

# gets the output path
output_path = os.path.join(dir_path, "chosen")


if os.path.exists(output_path):
	shutil.rmtree(output_path)

# creates the output path if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)



# resizes the input image to X by Y size and make them grayscale
def resize_grayscale(infile,x,y):
	filename, file_extension = os.path.splitext(infile)
	need_return = True
	if file_extension == ".png":
		need_return = False
	elif file_extension == ".jpeg":
		need_return = False
	elif file_extension == ".jpg":
		need_return = False
		
	if need_return:
		return
	
	print(infile) #prints filenames as it's processing it
	# reads the input image from the relative path
	image = cv2.imread(os.path.join(input_path, infile), 0)
	# resizes the image
	image = cv2.resize(image, (x, y))
	# writes the image to the relative new path
	cv2.imwrite(os.path.join(dir_path, "chosen", "{}.png".format(infilename)), image)




# Gets the files in the directory into an array
files_in_input_dir = os.listdir(input_path)

# makes sure infile and infilename are global variables.
infile = ""
infilename = ""


# goes through all the input images and resizes and saves them
iteration = 0
for i in files_in_input_dir:
	# gets the name of the current file with extension, eg. "test.png"
	infile = i

	# removes the extension, eg. "test.png" => "test"
	if use_file_name == True:
		infilename = infile[:-4]
	else:
		infilename = "{0}{1}".format(name_convension, iteration)
	
	# calls the resize function to resize the file. Resize_grayscale for grayscale output and resize for RGB output
	resize_grayscale(infile, x, y)
	iteration = iteration + 1
