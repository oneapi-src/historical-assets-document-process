# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring
# pylint: disable=E0401
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
from neural_compressor.experimental import Quantization, common
import crnn
import config
import mydataset
config.val_infofile = 'data/test.txt'
sampler = None
annot_list = []
cntr = 0
with open(config.val_infofile, encoding="utf-8") as f: # Read the images in the Test file 
    content = f.readlines()
    num_all = 0
    num_correct = 0
    for line in content: 
        if '\t' in line:
            fname, label = line.split('\t')
        else:
            fname, label = line.split('g:')
            fname += 'g'
        label = label.replace('\r', '').replace('\n', '')
        img = cv2.imread(fname) # Read the images
        w = 282
        h = 32
        imgH = config.imgH
        h, w = img.shape[:2]
        imgW = imgH * w // h
        transformer = mydataset.resizeNormalize((282, 32), is_test=True) # Resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Covert image from BGR to RGB
        img = Image.fromarray(np.uint8(img)).convert('L')
        img = transformer(img)
        img = img.view(1, *img.size())
        img = Variable(img) # Create the torch tensor
        if cntr < 1:
            image = img # Initializing the first image
        else:
            image = torch.cat((image, img), 0) # Image Concatenation
        annot_list.append(label) # Append label
        cntr = cntr + 1 # Increment counter

class Dataset:
    """Creating Dataset class for getting Image and labels"""
    def __init__(self):
        # init method or constructor
        test_images, test_labels = image, annot_list
        self.test_images = test_images
        self.labels = test_labels

    def __getitem__(self, index):
        """This function returns all the images and labels for the particular index
        input params:index
        output params:images and labels"""
        return self.test_images[index], self.labels[index]

    def __len__(self):
        """This function returns the length of images in the test dataset"""
        return len(self.test_images)

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m',
                        '--model_path_fp32',
                        type=str,
                        required=True,
                        help='path for fp32 model')

    parser.add_argument('-o',
                        '--output_path_int8',
                        type=str,
                        required=True,
                        help='output path for int8 model')
  
    parser.add_argument('-q',
                        '--quantization',
                        type=int,
                        required=False,
                        default=1,
                        help='0 for no quantization 1 for quantization')
    
    FLAGS = parser.parse_args()

    model_path = FLAGS.model_path_fp32
    output_path = FLAGS.output_path_int8 + "/"
    config_path = "config/conf.yaml"
    crnn = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh).to("cpu") # Initialize the model
    crnn.load_state_dict(torch.load(model_path, map_location="cpu")) # load the pretrained model
    dataset = Dataset() # Creating the Dataset object

    quantizer = Quantization(config_path) # Initialize the quantizer 
    quantizer.model = common.Model(crnn) # Load the crnn model to be quantized
    quantizer.calib_dataloader = common.DataLoader(dataset) # Load the data
    q_model = quantizer.fit() # quantize the model
    q_model.save(output_path) # save the model
