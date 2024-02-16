# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=pointless-string-statement
# pylint: disable=useless-import-alias
# pylint:disable=W0611,W0614,C0415,W0401,W0622
import cv2
from math import *
import numpy as np
from PIL import Image

import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def charRec(img, text_recs, model_path, quantized_model_path=None, intel_flag=False, inc_opt=False, adjust=False):
    """
    Recognize Text
    """
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]
    if inc_opt:
        import inc_inference as recognizer
        model, q_model = recognizer.load_model(model_path, quantized_model_path, intel_flag)
    else:    
        import inference as recognizer
        model = recognizer.load_model(model_path, intel_flag)

    total_inf_time = 0
    index = 0
    extracted_text = ""
    for rec in text_recs:
        xmin = rec[0] if rec[0]>0 else 0
        xmax = rec[1] if rec[1]>0 else 0
        ymin = rec[2] if rec[2]>0 else 0
        ymax = rec[3] if rec[3]>0 else 0
        partImg = img[ymin:ymax,xmin:xmax]
        
        print("Processing roi")
        Image.fromarray(partImg).save('./output/test_result/'+str(index)+".png")
        partImg = recognizer.preprocess_image(partImg)
        inf_time = 0
        if inc_opt:
            text, inf_time = recognizer.recognize_text(partImg, q_model)
        else:
            text, inf_time = recognizer.recognize_text(partImg, model)
        total_inf_time += inf_time
        text = text.strip()
        print("output from current roi: "+text)
        if len(text) > 0:
            results[index] = [rec]
            results[index].append(text)
            extracted_text = extracted_text + " " +text
        index+=1
    print("Extracetd text: \n"+extracted_text)
    return results, extracted_text, total_inf_time

def ocr(image, crnn_model_path, quantized_model_path=None, intel_flag=False, inc_opt=False):
    # detect
    bbox_list,free_list = reader.detect(image, width_ths=0)
    result, extracted_text, total_inf_time = charRec(image, bbox_list[0], crnn_model_path, quantized_model_path, intel_flag, inc_opt)
    return result, extracted_text, total_inf_time
