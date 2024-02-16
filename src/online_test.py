# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring
# pylint: disable=consider-using-with
# pylint: disable=E0401
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import utils
import keys
import config
import mydataset
alphabet = keys.alphabet_v2
converter = utils.strLabelConverter(alphabet.copy())

def val_model(infofile, model, gpu, log_file='0625.log'):
    '''This module validates the trained model
    input params:model,test data file
    output params:accuracy'''
    h = open('output/logs/{}'.format(log_file), 'w', encoding="utf-8")
    with open(infofile, encoding="utf-8") as f: # Read the file with test images list
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
            res = val_on_image(img, model, gpu) # Prediction on the image
            res = res.strip()
            label = label.strip()
            label = label.replace('"', '')
            if res == label: #  check if the predicted value is equal to the ground truth
                num_correct += 1 # increment the counter
            else:
                h.write('filename:{}\npred  :{}\ntarget:{}\n'.format(fname, res, label)) # print the wrong predicted images
            num_all += 1
    f.close()
    h.write('ocr_correct: {}/{}/{}\n'.format(num_correct, num_all, num_correct / num_all)) # Updating the log files
    #print(num_correct / num_all)
    h.close()
    return num_correct, num_all

def val_on_image(img, model, gpu):
    '''This modele performs the prediction on single image
    input params : image, model
    output params : predicted values'''

    imgH = config.imgH
    h, w = img.shape[:2]
    imgW = imgH * w // h

    transformer = mydataset.resizeNormalize((imgW, imgH), is_test=True) # resize the images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB
    image = Image.fromarray(np.uint8(img)).convert('L')
    image = transformer(image)
    if gpu:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image) # Convert the image to torch tensor

    model.eval() # Set the model to eval mode
    preds = model(image) # Perform prediction on the image

    preds = F.log_softmax(preds, 2)
    conf, preds = preds.max(2)  # Compute the Confidence score and the prediction 
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False) # Post processing
    return sim_pred


