####################
# Folder structure #
####################
./.git/, ./.idea/ and ./__pycashe__/ are automatically generated folders by the IDE and GitHub and can be ignored
./checkpoints/ is where TF models gets saved when you run the CNN.py
./resized/load/ is where the classify.py and NST.py gets its model and images from. Any image in the resized/load/ directory will be resized and moved to the chosen folder when run with classify.py
./resized/test/ is where the test data is. This data gets resized and augmented into the /augmented/ folder
./resized/train/ is where the train data is. This data gets resized and augmented into the /augmented/ folder
./resized/validate/ is where the validate data is. This data gets resized and augmented into the /augmented/ folder
./style_transfer/ this is the project folder for the Neural Style Transfer, and the /images/ contains the content and style images, and resized has the model and dataset
./test_data/ this is where the tests loads images from to use as referance. 
./tmp_img/ due to a problem with OpenCV and Matplotlib, I had to save the images to file before plotting them to get the color space correct, this is where they gets saved. 


####################
#   Python files   #
####################
./resized/load/resize_to_chosen.py
./style_transfer/NST.py the copy of the project file from Hvass Labs that I have edited to try to get working with style data
./style_transfer/style_data.py The dataobject for NST.py, to replace VGG16.py
./style_transfer/VGG16.py a local copy of the project file from Hvass Labs
./augment.py This is a library for augmenting images
./classify.py this is the classifier that works without training but rahter by loading a model
./CNN.py (CNN.pyc is compiled python) this is the trainer for the classifier
./data.py this is the data object for the CNN, hadles image loading and the likes
./run_cnn.py this runs the tests and then the CNN. Usefull for developing
./run_many.py this runs the CNN.py many times with many inputs
./tests.py the unit tests
./utils.py Used for plotting, collection of functions by Hvass Labs

####################
#   Other  files   #
####################

Goals.txt 
results.txt where the results of run_many.py gets saved. 
results.xlsx this is a collection of most of the results that have been gathered during this project.


####################
# Library Versions #
####################
Python3 		v. 3.5.2
	with GCC 	v. 5.4.0 20160609
	Pip3 		v. 10.0.1
TensorFlow		v. 1.5.0
Sci Learn		v. 0.19.1
OpenCV			v. 3.4.0
NumPy			v. 1.14.0
Matplotlib 		v. 2.1.2
Pillow			v. 5.0.0
