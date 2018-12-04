"""
package: Images2Dataset
class: DataAugmentation
Author: Rodrigo Loza
Description: Test class for data augmentation.
"""
# Libraries
import unittest
import math
import numpy as np
import cv2
from BoundingBoxAugmenters import *
from ImageAnnotation import * 
from VectorOperations import *

class BoundingBoxAugmenters_test(unittest.TestCase):
	
	def setUp(self):
		img_path = os.path.join(os.getcwd(), "tests/cars_dataset/images/cars0.png")
		annotation_path = os.path.join(os.getcwd(), "tests/cars_dataset/annotations/xmls/cars0.xml")
		assert os.path.isfile(img_path)
		assert os.path.isfile(annotation_path)
		# Image.
		self.frame = cv2.imread(img_path)
		self.annotation = annotation_path
		imgAnt = ImageAnnotation(path = self.annotation)
		self.bndboxes = imgAnt.propertyBoundingBoxes
		self.names = imgAnt.propertyNames
		# Augmenters.
		self.augmenter = BoundingBoxAugmenters()
		# Testing options.
		self.visualize = True
		self.waitTime = 1000
		self.windowSize = (800, 800)

	def tearDown(self):
		pass

	def test_scale(self):
		# Prepare data.
		frame = np.zeros([200, 200, 3])
		boundingBoxes = [[100, 100, 150, 150]]
		# Reduce test.
		size = (100, 100)
		scaled_frame, bndboxes = self.augmenter.scale(frame = frame,
																									boundingBoxes = boundingBoxes,
																									size = size,
																									interpolationMethod = 2)
		self.assertEqual(scaled_frame.shape[0], 100)
		self.assertEqual(scaled_frame.shape[1], 100)
		self.assertAlmostEqual(bndboxes[0][0], 50)
		self.assertAlmostEqual(bndboxes[0][1], 50)
		self.assertAlmostEqual(bndboxes[0][2], 75)
		self.assertAlmostEqual(bndboxes[0][3], 75)
		# Increase test
		boundingBoxes = [[100, 100, 150, 150]]
		size = (500, 500)
		scaled_frame, bndboxes = self.augmenter.scale(frame = frame,
																									boundingBoxes = boundingBoxes,
																									size = size,
																									interpolationMethod = 2)
		self.assertEqual(scaled_frame.shape[0], 500)
		self.assertEqual(scaled_frame.shape[1], 500)
		# Assert coordinates are equal for at least 10 units
		self.assertAlmostEqual(np.abs(bndboxes[0][0] - 250)//10, 0)
		self.assertAlmostEqual(np.abs(bndboxes[0][1] - 250)//10, 0)
		self.assertAlmostEqual(np.abs(bndboxes[0][2] - 375)//10, 0)
		self.assertAlmostEqual(np.abs(bndboxes[0][3] - 375)//10, 0)
		# Stretch test
		boundingBoxes = [[100, 100, 150, 150]]
		size = (400, 200)
		scaled_frame, bndboxes = self.augmenter.scale(frame = frame,
																									boundingBoxes = boundingBoxes,
																									size = size,
																									interpolationMethod = 2)
		self.assertEqual(scaled_frame.shape[0], 200)
		self.assertEqual(scaled_frame.shape[1], 400)
		self.assertAlmostEqual(bndboxes[0][0], 200)
		self.assertAlmostEqual(bndboxes[0][1], 100)
		self.assertAlmostEqual(bndboxes[0][2], 300)
		self.assertAlmostEqual(bndboxes[0][3], 150)
		# Visual test
		if (self.visualize):
			frame = self.frame.copy()
			scaled_frame, bndboxes = self.augmenter.scale(frame = frame,
																										boundingBoxes = self.bndboxes,
																										size = (140, 140),
																										interpolationMethod = 1)
			for each in bndboxes:
				ix, iy, x, y = each
				scaled_frame = cv2.rectangle(scaled_frame, (ix, iy), (x, y), (0,0,255), 3)
			cv2.namedWindow("__scaled__", 0)
			# cv2.resizeWindow("__scaled__", self.windowSize);
			cv2.imshow("__scaled__", scaled_frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_crop(self):
		# Prepare data.
		boundingBoxes = [[100, 100, 150, 150]]
		# Apply transformation.
		newboundingBoxes = self.augmenter.crop(boundingBoxes = boundingBoxes,
																				size = None)
		# print(boundingBoxes)
		# Assert values
		for i in range(len(newboundingBoxes)):
			ix, iy, x, y = newboundingBoxes[i]
			ixo, iyo, xo, yo = boundingBoxes[i]
			self.assertLess(x-ix, xo-ixo)
			self.assertLess(y-iy, yo-iyo)
		# Visual test.
		if (self.visualize):
			localbnxboxes = self.bndboxes
			frame = self.frame.copy()
			bndboxes = self.augmenter.crop(boundingBoxes = localbnxboxes,
																					size = (300,300))
			for each in bndboxes:
				ix, iy, x, y = each
				frame = cv2.rectangle(frame, (ix, iy), (x, y), (0,0,255), 5)
			cv2.namedWindow("__crop__", 0)
			# cv2.resizeWindow("__crop__", self.windowSize);
			cv2.imshow("__crop__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_pad(self):
		# Prepare data
		boundingBoxes = [[100, 100, 150, 150]]
		size = (50, 50)
		# Apply transformation
		boundingBoxes = self.augmenter.pad(boundingBoxes = boundingBoxes,
																							frameHeight = 200,
																							frameWidth = 200,
																							size = size)
		# print(boundingBoxes)
		self.assertLessEqual(boundingBoxes[0][0], 100)
		self.assertLessEqual(boundingBoxes[0][1], 100)
		self.assertGreaterEqual(boundingBoxes[0][2], 150)
		self.assertGreaterEqual(boundingBoxes[0][3], 150)
		# Visual test
		if (self.visualize):
			frame = self.frame.copy()
			bndboxes = self.augmenter.pad(frameHeight = self.frame.shape[0],
																			frameWidth = self.frame.shape[1],
																			boundingBoxes = self.bndboxes,
																			size = (25, 25))
			for each in bndboxes:
				ix, iy, x, y = each
				frame = cv2.rectangle(frame, (ix, iy), (x, y), (0,0,255), 2)
			cv2.namedWindow("__padding__", 0)
			# cv2.resizeWindow("__padding__", self.windowSize);
			cv2.imshow("__padding__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()
	
	def test_jitter_boxes(self):
		if (self.visualize):
			frame = self.augmenter.jitterBoxes(frame = self.frame,
																					boundingBoxes = self.bndboxes,
																					size = (20,20),
																					quantity = 20)
			cv2.namedWindow("__jitterBoxes__", 0)
			# cv2.resizeWindow("__jitterBoxes__", self.windowSize);
			cv2.imshow("__jitterBoxes__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_horizontal_flip(self):
		if (self.visualize):
			# Perform horizontal flip.
			frame = self.augmenter.horizontalFlip(frame = self.frame.copy(),
																						boundingBoxes = self.bndboxes)
			# Visualization.
			cv2.namedWindow("__horizontalFlip__", 0)
			# cv2.resizeWindow("__horizontalFlip__", self.windowSize);
			cv2.imshow("__horizontalFlip__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_vertical_flip(self):
		if (self.visualize):
			# Perform vertical flip.
			frame = self.augmenter.verticalFlip(frame = self.frame.copy(),
																						boundingBoxes = self.bndboxes)
			# Visualization.
			cv2.namedWindow("__verticalFlip__", 0)
			# cv2.resizeWindow("__verticalFlip__", self.windowSize);
			cv2.imshow("__verticalFlip__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_rotation(self):
		if (self.visualize):
			theta = 0.0
			for i in range(10):
				frame = self.augmenter.rotation(frame = self.frame, 
																					boundingBoxes = self.bndboxes,
																					theta = theta)
				theta += 0.3
				cv2.namedWindow("__rotation__", 0)
				# cv2.resizeWindow("__rotation__", self.windowSize);
				cv2.imshow("__rotation__", frame)
				cv2.waitKey(250)
				cv2.destroyAllWindows()

	def test_dropout(self):
		if (self.visualize):
			frame = self.augmenter.dropout(frame = self.frame,
												              boundingBoxes = self.bndboxes,
												              size = (25,25),
												              threshold = 0.5)
			cv2.namedWindow("__dropout__", 0)
			# cv2.resizeWindow("__dropout__", self.windowSize);
			cv2.imshow("__dropout__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_rotation_equations(self):
		x, y, theta = 10, 10, math.pi / 2
		x, y = VectorOperations.rotation_equations(x, y, theta)
		self.assertEqual(x, -10)
		self.assertEqual(y, 10)

if __name__ == "__main__":
	unittest.main()
