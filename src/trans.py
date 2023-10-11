# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python
#coding:utf-8
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=super-with-arguments
# pylint: disable=no-else-return
# pylint: disable=arguments-renamed
# pylint: disable=E0401,W0311
import random
import abc
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageEnhance, ImageFilter
import cv2
import trans_utils
from trans_utils import getpilimage

index = 0
class TransBase():
	# Create a class for translation
	def __init__(self, probability=1.):
		# init method or constructor
		super(TransBase, self).__init__()
		self.probability = probability
	
	@abc.abstractmethod
	def tranfun(self, inputimage):
		pass

	def process(self, inputimage):
		""" This function checks if the probability value is less than the random values generated
		input params: inputimage
		output params: transformed image """

		if np.random.random() < self.probability:
			return self.tranfun(inputimage)
		else:
			return inputimage

class RandomContrast(TransBase):
	# Create a class for contrast enhancement on the images 

	def setparam(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."  # nosec
		assert self.lower >= 0, "lower must be non-negative."  # nosec
	
	def tranfun(self, image):
		image = getpilimage(image)
		enh_con = ImageEnhance.Brightness(image)
		return enh_con.enhance(random.uniform(self.lower, self.upper))  # nosec

class RandomBrightness(TransBase):
	 # Create a class for Brightness enhancement on the images
	def setparam(self, lower=0.5, upper=1.5):
		""" This function sets the brightness limits 
		input params : lower and upper thresholds
		"""
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."  # nosec
		assert self.lower >= 0, "lower must be non-negative."  # nosec
	
	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		image = getpilimage(image)
		bri = ImageEnhance.Brightness(image)
		return bri.enhance(random.uniform(self.lower, self.upper))  # nosec

class RandomColor(TransBase):
	# Create a class for Random color changes on the images

	def setparam(self, lower=0.5, upper=1.5):
		""" This function sets the color level thresholds
		input params : lower and upper thresholds
		"""
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."  # nosec
		assert self.lower >= 0, "lower must be non-negative."  # nosec
	
	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """

		image = getpilimage(image)
		col = ImageEnhance.Color(image)
		return col.enhance(random.uniform(self.lower, self.upper))  # nosec

class RandomSharpness(TransBase):
	# Create class to enhance Sharpness
	def setparam(self, lower=0.5, upper=1.5):
		""" This function sets the color level thresholds
		input params : lower and upper thresholds
		"""
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."  # nosec
		assert self.lower >= 0, "lower must be non-negative."  # nosec
	
	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		image = getpilimage(image)
		sha = ImageEnhance.Sharpness(image)
		return sha.enhance(random.uniform(self.lower, self.upper))  # nosec

class Compress(TransBase):
# Create class for image compression
	def setparam(self, lower=5, upper=85):
		""" This function sets the compression level thresholds
		input params : lower and upper thresholds
		"""
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."  # nosec
		assert self.lower >= 0, "lower must be non-negative."  # nosec
	
	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		img = trans_utils.getcvimage(image)
		param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(self.lower, self.upper)]  # nosec
		img_encode = cv2.imencode('.jpeg', img, param)
		img_decode = cv2.imdecode(img_encode[1], cv2.IMREAD_COLOR)
		pil_img = trans_utils.cv2pil(img_decode)
		if len(image.split()) == 1:
			pil_img = pil_img.convert('L')
		return pil_img

class Exposure(TransBase):
  # Creates the class for window transformations  
	def setparam(self, lower=5, upper=10):
		""" This function sets the  thresholds
		input params : lower and upper thresholds
		"""
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."  # nosec
		assert self.lower >= 0, "lower must be non-negative."  # nosec
	
	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		image = trans_utils.getcvimage(image)
		h, w = image.shape[:2]
		x0 = random.randint(0, w)  # nosec
		y0 = random.randint(0, h)  # nosec
		x1 = random.randint(x0, w)  # nosec
		y1 = random.randint(y0, h)  # nosec
		transparent_area = (x0, y0, x1, y1)
		mask = Image.new('L', (w, h), color=255)
		draw = ImageDraw.Draw(mask)
		mask = np.array(mask)
		if len(image.shape) == 3:
			mask = mask[:, :, np.newaxis]
			mask = np.concatenate([mask, mask, mask], axis=2)
		draw.rectangle(transparent_area, fill=random.randint(150, 255))  # nosec
		reflection_result = image + (255 - mask)
		reflection_result = np.clip(reflection_result, 0, 255)
		return trans_utils.cv2pil(reflection_result)

class Rotate(TransBase):
# Create class for image rotation
	def setparam(self, lower=-5, upper=5):
		""" This function sets the  thresholds 
		input params : lower and upper thresholds
		"""
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."  # nosec
		# assert self.lower >= 0, "lower must be non-negative."
	
	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		image = getpilimage(image)
		rot = random.uniform(self.lower, self.upper)  # nosec
		trans_img = image.rotate(rot, expand=True)
		# trans_img.show()
		return trans_img

class Blur(TransBase):
# Create class for image blurr
	def setparam(self, lower=0, upper=1):
		""" This function sets the  thresholds
		input params : lower and upper thresholds
		"""
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."  # nosec
		assert self.lower >= 0, "lower must be non-negative."  # nosec
	
	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		image = getpilimage(image)
		image = image.filter(ImageFilter.GaussianBlur(radius=1))
		# blurred_image = image.filter(ImageFilter.Kernel((3,3), (1,1,1,0,0,0,2,0,2)))
		# Kernel
		return image

class Salt(TransBase):
# Create class for Noise addition
	def setparam(self, rate=0.02):
		""" This function sets the  thresholds
		input params : lower and upper thresholds
		"""
		self.rate = rate

	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		image = getpilimage(image)
		num_noise = int(image.size[1] * image.size[0] * self.rate)
		# assert len(image.split()) == 1
		for k in range(num_noise):
			i = int(np.random.random() * image.size[1])
			j = int(np.random.random() * image.size[0])
			image.putpixel((j, i), int(np.random.random() * 255))
		return image

class AdjustResolution(TransBase):
# Create class for changing resolution
	def setparam(self, max_rate=0.95, min_rate=0.5):
		""" This function sets the  thresholds
		input params : lower and upper thresholds
		"""
		self.max_rate = max_rate
		self.min_rate = min_rate

	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		image = getpilimage(image)
		w, h = image.size
		rate = np.random.random()*(self.max_rate-self.min_rate)+self.min_rate
		w2 = int(w*rate)
		h2 = int(h*rate)
		image = image.resize((w2, h2))
		image = image.resize((w, h))
		return image


class Crop(TransBase):
# Create class for image cropping
	def setparam(self, maxv=2):
		""" This function sets the  max values
		input params : max value
		"""
		self.maxv = maxv

	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		img = trans_utils.getcvimage(image)
		h, w = img.shape[:2]
		org = np.array([[0, np.random.randint(0, self.maxv)],
						[w, np.random.randint(0, self.maxv)],
						[0, h-np.random.randint(0, self.maxv)],
						[w, h-np.random.randint(0, self.maxv)]], np.float32)
		dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
		M = cv2.getPerspectiveTransform(org, dst)
		res = cv2.warpPerspective(img, M, (w, h))
		return getpilimage(res)

class Crop2(TransBase):
# Create class for image cropping
	def setparam(self, maxv_h=4, maxv_w=4):
		""" This function sets the  thresholds
		input params : lower and upper thresholds
		"""
		self.maxv_h = maxv_h
		self.maxv_w = maxv_w

	def tranfun(self, image_and_loc):
		"""This function performs the transform operation on the image 
		input params : image """
		image, left, top, right, bottom = image_and_loc
		w, h = image.size
		left = np.clip(left, 0, w-1)
		right = np.clip(right, 0, w-1)
		top = np.clip(top, 0, h-1)
		bottom = np.clip(bottom, 0, h-1)
		img = trans_utils.getcvimage(image)
		try:
			# global index
			res = getpilimage(img[top:bottom, left:right])
			# res.save('test_imgs/crop-debug-{}.jpg'.format(index))
			# index+=1
			return res

		except AttributeError as e:
			print(e)
			image.save('test_imgs/t.png')
			print(left, top, right, bottom)
		h = bottom - top
		w = right - left
		org = np.array([[left - np.random.randint(0, self.maxv_w), top + np.random.randint(-self.maxv_h, self.maxv_h//2)],
						[right + np.random.randint(0, self.maxv_w), top + np.random.randint(-self.maxv_h, self.maxv_h//2)],
						[left - np.random.randint(0, self.maxv_w), bottom - np.random.randint(-self.maxv_h, self.maxv_h//2)],
						[right + np.random.randint(0, self.maxv_w), bottom - np.random.randint(-self.maxv_h, self.maxv_h//2)]], np.float32)
		dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
		M = cv2.getPerspectiveTransform(org, dst)
		res = cv2.warpPerspective(img, M, (w, h))
		return getpilimage(res)

class Stretch(TransBase):
# Create class for image stretching
	def setparam(self, max_rate=1.2, min_rate=0.8):
		""" This function sets the  thresholds
		input params : lower and upper thresholds
		"""
		self.max_rate = max_rate
		self.min_rate = min_rate

	def tranfun(self, image):
		"""This function performs the transform operation on the image 
		input params : image """
		image = getpilimage(image)
		w, h = image.size
		rate = np.random.random() * (self.max_rate - self.min_rate) + self.min_rate
		w2 = int(w * rate)
		image = image.resize((w2, h))
		return image


