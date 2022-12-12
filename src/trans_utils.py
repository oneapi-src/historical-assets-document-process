# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=no-else-return
# pylint: disable=unused-variable
# pylint: disable=inconsistent-return-statements
# pylint: disable=import-error
# pylint: disable=W0311
import os
import shutil
import time
import datetime
import hashlib
import cv2
import numpy as np
import PIL
from PIL import Image

def rename(filepath):
	""" This function renames the image in the filepath 
	input params : filepath"""
	filelist = os.listdir(filepath)
	filelist.sort()
	i = 1
	for filename in filelist:
		if str(filename) == '.DS_Store':
				continue
		ext = filename.split('.')[-1]
		shutil.move(filepath + '/' + filename, filepath + '/' + str(i).zfill(3) + '.' + ext)
		i += 1

def zlog(func):
	""" This function  calculates the time """
	def new_fn(*args):
		""" This function  calculates the time"""
		start = time.time()
		result = func(*args)
		end = time.time()
		duration = end - start
		duration = "%.4f" % duration
		fulltime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
		return result
	return new_fn


def getpilimage(image):
	""" This function reads the image  using pil library
	Input params :image"""

	if isinstance(image, PIL.Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
		return image
	elif isinstance(image, np.ndarray):
		return cv2pil(image)


def getcvimage(image):
	""" This function reads the image  using opencv
	Input params :image"""

	if isinstance(image, np.ndarray):
		return image
	elif isinstance(image, PIL.Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
		return pil2cv(image)


def cshowone(image):
	""" This function displays the image 
	Input params :image"""

	image = getcvimage(image)
	cv2.imshow('tmp', image)
	cv2.waitKey(3000)
	#return


def pshowone(image):
	""" This function displays the image 
	Input params :image"""

	image = getpilimage(image)
	image.show()
	#return


def cshowtwo(image1, image2):
	""" This function stitches two images into one image
	input params : images to be stitched 
	output params : Single image"""
	width = 800 / 2
	height = 500 / 2
	image1 = getpilimage(image1)
	image2 = getpilimage(image2)
	h, w = image1.size
	image1 = image1.resize((int(width), int(h * height / w)))
	image2 = image2.resize(image1.size)
	bigimg = Image.new('RGB', (width * 2, image1.size[1]))

	bigimg.paste(image1, (0, 0, image1.size[0], image1.size[1]))
	bigimg.paste(image2, (width, 0, width + image1.size[0], image1.size[1]))
	bigimg = getcvimage(bigimg)
	cshowone(bigimg)
	#return


def pshowtwo(image1, image2):
	""" This function stitches two images into one image
	input params : images to be stitched 
	output params : Single image"""

	width = int(800 / 2)
	height = int(500 / 2)
	image1 = getpilimage(image1) # Read Image1
	image2 = getpilimage(image2) # Read Image1
	h, w = image1.size
	image1 = image1.resize((int(width), int(h * height / w))) 
	image2 = image2.resize(image1.size)
	bigimg = Image.new('RGB', (width * 2, image1.size[1]))

	bigimg.paste(image1, (0, 0, image1.size[0], image1.size[1]))
	bigimg.paste(image2, (width, 0, width + image1.size[0], image1.size[1]))
	pshowone(bigimg)
	#return

def pil2cv(image):
	"""This function converts the image from pil to cv
	input params :image
	output params :image"""

	if len(image.split()) == 1:
		return np.asarray(image)
	elif len(image.split()) == 3:
		return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
	elif len(image.split()) == 4:
		return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)


def cv2pil(image):
	"""This function converts the image from cv2 format to ndarray
	input param :image 
	output param :converted image"""

	assert isinstance(image, np.ndarray), 'input image type is not cv2'  # nosec
	if len(image.shape) == 2:
		return Image.fromarray(image)
	elif len(image.shape) == 3:
		return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def rgb2gray(filename):
	"""This function converts the rgb image to gray 
	input params: image path 
	output :gray scale image"""

	im = Image.open(filename).convert('L')
	im.show()

	new_image = Image.new("L", (im.width + 6, im.height + 6), 0)
	out_image = Image.new("L", (im.width + 6, im.height + 6), 0)

	new_image.paste(im, (3, 3, im.width + 3, im.height + 3))

	im = getcvimage(im)
	new_image = getcvimage(new_image)
	out_image = getcvimage(out_image)

	_, thresh = cv2.threshold(new_image, 0, 255, cv2.THRESH_OTSU)
	pshowone(thresh)
	image, contours, hierarchy = cv2.findContours(thresh, 3, 2)
	cv2.polylines(out_image, contours, True, 255)
	image = getpilimage(out_image)
	im = getpilimage(im)
	image = image.crop((3, 3, im.width + 3, im.height + 3))
	image.show()
	return image