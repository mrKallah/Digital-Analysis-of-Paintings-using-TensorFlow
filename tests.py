import unittest


import augment as aug

import data as data

from data import *


class TestStringMethods(unittest.TestCase):

	def test_bright_blur(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.add_light_and_color_blur(img)

		load = cv2.imread("test_data/augmented/bright_blur.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_affine(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.affine_transform(img, 3)

		load = cv2.imread("test_data/augmented/affine.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_darken(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.brighten_darken(img, 0.6)

		load = cv2.imread("test_data/augmented/darken.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_brighten(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		load = cv2.imread("test_data/augmented/brighten.png", 3)

		img = aug.brighten_darken(img, 1.4)

		np.testing.assert_array_equal(img, load)

	def test_blur(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.blur(img)

		load = cv2.imread("test_data/augmented/blur.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_sharpen(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.sharpen(img)

		load = cv2.imread("test_data/augmented/sharpen.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_rotate(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.rotate(img, -2)

		load = cv2.imread("test_data/augmented/rotate.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_crop(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.crop(img, 1, 0, 0, 0)

		load = cv2.imread("test_data/augmented/crop.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_mirror(self):
		img = cv2.imread("test_data/augmented/org.png", 3)
		img = aug.mirror(img)

		load = cv2.imread("test_data/augmented/mirror.png", 3)

		np.testing.assert_array_equal(img, load)

	def test_one_hot_encode(self):
		labels = [0, 0, 1, 1]
		encoded = one_hot_encode(labels)
		true = [[0, 1], [0, 1], [1, 0], [1, 0]]
		np.testing.assert_array_equal(encoded, true)

	def test_iterate(self):
		aug.iterate()
		aug.iterate()
		aug.iterate()
		aug.iterate()
		aug.iterate()
		aug.iterate()
		self.assertEqual(aug.iterate(), 6)

	def test_load_data(self):
		images, labels = load_data("test_data", "bright_blur", 500 * 500, 3, file_format="png")

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
		arr = arr.reshape(-1, 500 * 500)

		np.testing.assert_array_equal(labels, [1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
		np.testing.assert_array_equal(arr[2], images[2])

	def test_next_data_batch(self):
		# next_batch(self, batch_size, shuffle=True)

		data.train.images, data.train.labels = load_data("test_data", "sharpen", 500 * 500, 3, file_format="png")
		data.train = data.train.init()

		batch_images, batch_labels = next_batch(data.train, 3, shuffle=False)

		image_selection = [data.train.images[0], data.train.images[1], data.train.images[2]]
		image_selection = np.asarray(image_selection)

		label_selection = [data.train.labels[0], data.train.labels[1], data.train.labels[2]]
		label_selection = np.asarray(label_selection)

		np.testing.assert_array_equal(image_selection, batch_images)
		np.testing.assert_array_equal(label_selection, batch_labels)


if __name__ == '__main__':
	unittest.main()
