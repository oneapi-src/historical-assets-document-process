# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=useless-import-alias
# pylint: disable=import-error
import argparse
import string
import time
import numpy as np
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import crnn as crnn
import config
import mydataset
import keys
import utils
import crnn

alphabet = keys.alphabet_v2
converter = utils.strLabelConverter(alphabet.copy())
config.val_infofile = 'data/test.txt'
#assert test_dataset
sampler = None
images_list = []
annot_list = []
cntr = 0

#Model Evaulation
def evaluate_model(model):
    '''This module evaluates the model performance of the model and calculates the accuracy
       @input params: model
       @output :accuracy
    '''
    number_of_correct = 0
    total = len(images_list)
    for i in range(0, len(images_list)):
        sim_pred =""
        preds = model(images_list[i])

        sim_pred = process_preds(preds)
        
        annot_list[i] = annot_list[i].replace('"', ' ')
        sim_pred = sim_pred.replace('"', ' ')
        #print("predicted:", sim_pred)
        #print("targeted:", annot_list[i])
        if sim_pred.strip() == annot_list[i].strip():
            number_of_correct = number_of_correct+1
        #else:
        #    print("predicted:", sim_pred)
        #    print("targeted:", annot_list[i])
    accuracy = number_of_correct/total
    return accuracy

#Post Process Model Predictions & decode to text
def process_preds(preds):
    preds = F.log_softmax(preds, 2)
    conf, preds = preds.max(2)
    
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred

#Pre Process the image for inference
def preprocess_image(img):
    h,w = img.shape[:2]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(img)
    transformer = mydataset.resizeNormalize((int(w/h*32), 32))
    image = transformer(image)
    image = image.view(1, *image.size())
    image = Variable(image)
    
    return image

#Pre-Process Inference Dataset
def process_test_dataset(batchsize):
    with open(config.val_infofile, encoding="utf-8") as f:
        content = f.readlines()
        num_all = 0
        num_correct = 0
        cntr = 0
        i = 1
        for line in content:
            if i<=batchsize:
                i+=1
                #import pdb;pdb.set_trace()
                if '\t' in line:
                    fname, label = line.split('\t')
                else:
                    fname, label = line.split('g:')
                    fname += 'g'
                label = label.replace('\r', '').replace('\n', '')
                img = cv2.imread(fname)
                #print(img.shape)
                #img = preprocess_image(img)
                # w = 282
                h = 32
                imgH = config.imgH
                h, w = img.shape[:2]
                imgW = imgH * w//h
                transformer = mydataset.resizeNormalize((282, 32), is_test=True)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(np.uint8(img)).convert('L')
                img = transformer(img)
                
                img = img.view(1, *img.size())
                img = Variable(img)
                #print(img.shape)
                if cntr < 1:
                    image = img
                else:
                    image = torch.cat((image, img), 0)
                
                images_list.append(img)
                annot_list.append(label)
                cntr = cntr + 1
            else:
                break
    return image

#Inference on Image 
def recognize_text(image, crnn_model):
    total_time = 0
    for i in range(20):
        start = time.time()
        preds=crnn_model(image)
        total_time += time.time()-start
    print("Average Batch Inference time taken for in seconds ---> ", total_time/20)
    inf_time = total_time/20
    text = process_preds(preds)
    return text, inf_time

#Load CRNN Model
def load_model(model_path, intel_flag):
    crnn_model = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh)
    crnn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    crnn_model.eval()

    if intel_flag:
        import intel_extension_for_pytorch as ipex  # pylint: disable=E0401
        crnn_model = ipex.optimize(crnn_model)
        print("Intel Pytorch Optimizations has been Enabled!")
    else:
        device = torch.device('cpu')
    return crnn_model


if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 for enabling intel pytorch optimizations, default is 0')

    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        default="",
                        help='path for model')

    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=100,
                        help='batch size for inferencing')
                        
    FLAGS = parser.parse_args()
    intel_flag = FLAGS.intel
    model_path = FLAGS.model_path
    batchsize = FLAGS.batchsize

    #Load Model
    model = load_model(model_path, intel_flag)

    #Process Test Dataset
    image = process_test_dataset(batchsize)

    #Text Recognition
    recognize_text(image, model)
    
    #Evaluate model accuracy
    accuracy = evaluate_model(model)
    print("Accuracy is --->",accuracy)
