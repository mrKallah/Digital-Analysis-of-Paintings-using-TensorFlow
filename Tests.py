import unittest

import Augment as aug

from Data import *


def get_classes():
	return ["rand", "gogh", "bernard", "bonnard", "ibels", "kunisada", "kw"]

class TestStringMethods(unittest.TestCase):


	def test_augmented_not_same_as_original(self):
		"""
		Tests that the augmented images are not the same as the original
		"""
		load = cv2.imread("test_data/augmented/org.png", 3)
		org = cv2.imread("test_data/augmented/org.png", 3)

		#make sure my method of testing is valid
		img = org
		assert np.ndarray.tolist(img) == np.ndarray.tolist(load)

		img = aug.affine_transform(org)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.add_light_and_color_blur(img)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.affine_transform(org)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.brighten_darken(org, 0.6)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.brighten_darken(org, 1.4)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.blur(org)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.sharpen(org)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.rotate(org, -2)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.crop(org, 1, 0, 0, 0)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

		img = aug.mirror(org)
		assert np.ndarray.tolist(img) != np.ndarray.tolist(load)

	def test_that_wrong_augment_swtich_paramaters_errors(self):
		"""
		Tests that the augment switch returns -1 if the inputs are wrong
		"""
		img = np.array([[1,1,1],[2,2,2],[3,3,3]])
		test = aug.augment_switch(img, "test_data/tmp/", "3x3", 1, 3, print_errors=False)
		assert test == -1

	def test_resize(self):
		"""
		Tests that the resize augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.resize(img, (50, 50))

		np.testing.assert_array_equal(img.shape, (50, 50, 3))

	def test_bright_blur(self):
		"""
		Tests that the bright blur augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.add_light_and_color_blur(img)

		load = cv2.imread("test_data/augmented/bright_blur.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_affine(self):
		"""
		Tests that the affine transformation augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.affine_transform(img)

		load = cv2.imread("test_data/augmented/affine.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_darken(self):
		"""
		Tests that the darken augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.brighten_darken(img, 0.6)

		load = cv2.imread("test_data/augmented/darken.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_brighten(self):
		"""
		Tests that the brighten augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		load = cv2.imread("test_data/augmented/brighten.png", 3)

		img = aug.brighten_darken(img, 1.4)

		np.testing.assert_array_equal(img, load)

	def test_blur(self):
		"""
		Tests that the blur augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.blur(img)

		load = cv2.imread("test_data/augmented/blur.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_sharpen(self):
		"""
		Tests that the sharpen augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.sharpen(img)

		load = cv2.imread("test_data/augmented/sharpen.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_rotate(self):
		"""
		Tests that the rotate augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.rotate(img, -2)

		load = cv2.imread("test_data/augmented/rotate.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_crop(self):
		"""
		Tests that the crop augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.crop(img, 1, 0, 0, 0)

		load = cv2.imread("test_data/augmented/crop.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_mirror(self):
		"""
		Tests that the mirror augmentation has not changed
		"""
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.mirror(img)

		load = cv2.imread("test_data/augmented/mirror.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_one_hot_encode(self):
		"""
		Tests that the one_hot_encoder works how you would expect it to.
		"""
		labels = ["gogh", "rand", "kw", "kunisada"]

		encoded = one_hot_encode(labels, get_classes(), ".png")
		true = [[0, 1, 0, 0, 0, 0, 0],
			   [1, 0, 0, 0, 0, 0, 0],
			   [0, 0, 0, 0, 0, 0, 1],
			   [0, 0, 0, 0, 0, 1, 0]]
		np.testing.assert_array_equal(encoded, true)

	def test_iterate(self):
		"""
		Tests that the iteration and re-iteration functions works as expected
		"""
		aug.re_iterate()
		aug.iterate()
		aug.iterate()
		aug.iterate()
		aug.iterate()
		aug.iterate()
		aug.iterate()
		self.assertEqual(aug.iterate(), 6)
		aug.re_iterate()
		aug.iterate()
		self.assertEqual(aug.iterate(), 1)
		self.assertEqual(aug.iterate(6, True), 7)

	def test_load_data(self):
		"""
		Tests that the load data function loads the data in the correct way and expected way
		"""

		images, labels = load_data("test_data", 500 * 500, 3)
		# [rotate, brighten, crop, mirror, blur, sharpen, bright_blur, org, affine, darken]


		brighten = cv2.imread("test_data/augmented/brighten.png", 3)
		brighten = np.asarray(brighten, dtype="float32")

		rotate = cv2.imread("test_data/augmented/rotate.png", 3)
		rotate = np.asarray(rotate, dtype="float32")

		blur = cv2.imread("test_data/augmented/blur.png", 3)
		blur = np.asarray(blur, dtype="float32")

		crop = cv2.imread("test_data/augmented/crop.png", 3)
		crop = np.asarray(crop, dtype="float32")

		mirror = cv2.imread("test_data/augmented/mirror.png", 3)
		mirror = np.asarray(mirror, dtype="float32")

		sharpen = cv2.imread("test_data/augmented/sharpen.png", 3)
		sharpen = np.asarray(sharpen, dtype="float32")

		bright_blur = cv2.imread("test_data/augmented/bright_blur.png", 3)
		bright_blur = np.asarray(bright_blur, dtype="float32")

		org = cv2.imread("test_data/augmented/org.png", 3)
		org = np.asarray(org, dtype="float32")

		affine = cv2.imread("test_data/augmented/affine.png", 3)
		affine = np.asarray(affine, dtype="float32")

		darken = cv2.imread("test_data/augmented/darken.png", 3)
		darken = np.asarray(darken, dtype="float32")

		# [rotate, brighten, crop, mirror, blur, sharpen, bright_blur, org, affine, darken]
		arr = []

		arr.append(rotate)
		arr.append(brighten)
		arr.append(crop)
		arr.append(mirror)
		arr.append(blur)
		arr.append(sharpen)
		arr.append(bright_blur)
		arr.append(org)
		arr.append(affine)
		arr.append(darken)

		arr = np.asarray(arr)
		arr = arr.reshape(-1, 500 * 500, 3)

		# [rotate, brighten, crop, mirror, blur, sharpen, bright_blur, org, affine, darken]

		test =  np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
		np.testing.assert_array_equal(labels.shape, test.shape)
		np.testing.assert_array_equal(arr.shape, images.shape)

	def test_next_data_batch(self):
		"""
		Tests that the next_batch function works
		"""
		# next_batch(self, batch_size, shuffle=True)

		data.train.images, data.train.labels = load_data("test_data", 500 * 500, 3)


		data.train = data.train.init(['mirror', 'sharpen', 'rotate', 'brighten', 'affine', 'blur', 'brightblur', 'crop', 'darken', 'org'])

		batch_images, batch_labels = next_batch(data.train, 3, shuffle=False, test=True)

		image_selection = [data.train.images[0], data.train.images[1], data.train.images[2]]
		image_selection = np.asarray(image_selection)

		label_selection = [data.train.labels[0], data.train.labels[1], data.train.labels[2]]
		label_selection = np.asarray(label_selection)

		np.testing.assert_array_equal(image_selection, batch_images)
		np.testing.assert_array_equal(label_selection, batch_labels)



if __name__ == '__main__':
	unittest.main()
