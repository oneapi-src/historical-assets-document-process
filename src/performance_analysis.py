# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=E0401
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2  # pylint: disable=E0401
from PIL import Image
import intel_extension_for_pytorch as ipex
from neural_compressor.utils.pytorch import load
import config
import mydataset
import keys
import utils
import crnn

alphabet = keys.alphabet_v2
converter = utils.strLabelConverter(alphabet.copy())
config.val_infofile = 'data/test.txt'
sampler = None
images_list = []
annot_list = []
cntr = 0

with open(config.val_infofile, encoding="utf-8") as f: # Reading all the images from the test file
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
        img = cv2.imread(fname) # Read image 
        w = 282
        h = 32
        imgH = config.imgH
        h, w = img.shape[:2]
        imgW = imgH * w//h
        transformer = mydataset.resizeNormalize((282, 32), is_test=True) # resize the images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image from BGR to RGB
        img = Image.fromarray(np.uint8(img)).convert('L')
        img = transformer(img)
        img = img.view(1, *img.size())
        img = Variable(img) # Create a torch tensor
        images_list.append(img) # append images
        annot_list.append(label) # append labels
        cntr = cntr+1 # Increment counter

def evaluate_model(model):
    '''This module evaluates the model performance of the model and calculates the accuracy
       input params: model
       output :accuracy
    '''
    number_of_correct = 0
    total = len(images_list)
    for i in range(0, len(images_list)): # looping through the images
        preds = model(images_list[i]) # Perform prediction on the image list
        preds = F.log_softmax(preds, 2)
        conf, preds = preds.max(2) # Compute the confidence and prediction values
        # Post processing of Predictions
        preds = preds.transpose(1, 0).contiguous().view(-1) 
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False) # Get the prediction label
        annot_list[i] = annot_list[i].replace('"', ' ')
        sim_pred = sim_pred.replace('"', ' ') 
        print("predicted:", sim_pred)
        print("targeted:", annot_list[i])
        if sim_pred.strip() == annot_list[i].strip(): # Compare the Predicted string and targeted string
            number_of_correct = number_of_correct+1 # if matches increment
    accuracy = number_of_correct/total # Compute accuracy
    return accuracy

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path_fp32',
                        type=str,
                        required=True,
                        default="",
                        help='path for fp32 model')
    parser.add_argument('-q',
                        '--model_path_int8',
                        type=str,
                        required=True,
                        default="",
                        help='path for int8 model')
    FLAGS = parser.parse_args()
    model_path = FLAGS.model_path_fp32
    crnn = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh) # Initialize model
    crnn.load_state_dict(torch.load(model_path, map_location="cpu")) # Load the pretrained model
    quantized_model_path = FLAGS.model_path_int8
    int8_model = load(quantized_model_path, crnn) # Load the model
    crnn.eval() # Set the model to the eval path
    crnn = ipex.optimize(crnn) # optimize the  FP32 model using ipex
    int8_model = ipex.optimize(int8_model) # optimize the  INT8 model using ipex
    #  EVALUATION
    print("*" * 50)
    print("Evaluating the FP32 Model")
    print("*" * 50)
    accuracy_fp32 = evaluate_model(crnn)
    print("Accuracy of FP32 model :", accuracy_fp32)
    print("*" * 50)
    print("Evaluating the INT8 Model")
    print("*" * 50)
    accuracy_int8 = evaluate_model(int8_model)
    print("Accuracy of INT8 model :", accuracy_int8)
