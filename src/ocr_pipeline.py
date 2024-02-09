# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=pointless-string-statement
# pylint: disable=useless-import-alias
# pylint: disable=W0311,E0401,W0622,W0612,W0105,C0411,W1514
import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import argparse

#Process OCR on single image
def single_pic_proc(image_file, crnn_model_path, quantized_model_path=None, intel_opt=False):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, extracted_text, total_inf_time = ocr(image, crnn_model_path, quantized_model_path, intel_opt, inc_opt)
    return result, extracted_text, total_inf_time


if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--test_dataset_path',
                        type=str,
                        required=True,
                        default="",
                        help='Test Images Path')

    parser.add_argument('-q',
                        '--inc',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 for enabling INC quantized model for inferencing, default is 0')

    parser.add_argument('-m',
                        '--crnn_model_path',
                        type=str,
                        required=True,
                        default="",
                        help='path of Fp32 CRNN model')

    parser.add_argument('-n',
                        '--quantized_model_path',
                        type=str,
                        required=False,
                        default="",
                        help='path of Quantized Int8 model, If INC is tru then it must be given')

    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=100,
                        help='batch size for inferencing')
                        
    FLAGS = parser.parse_args()
    intel_opt = 1
    inc_opt = FLAGS.inc
    crnn_model_path = FLAGS.crnn_model_path
    quantized_model_path = FLAGS.quantized_model_path
    batch_size = FLAGS.batch_size
    test_images_path = FLAGS.test_dataset_path

    if inc_opt:
        intel_opt = True
        if (quantized_model_path is None): raise AssertionError('There is not a quantized model') 

    image_files = glob(test_images_path+'/*.*')
    print(image_files)
    result_dir = './output/test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    total_e2e_time = 0
    for image_file in sorted(image_files):
        t = time.time()
        result, extracted_text, total_inf_time = single_pic_proc(image_file, crnn_model_path, quantized_model_path, intel_opt)
        total_e2e_time += total_inf_time
        print("Prediction time for image: ", total_inf_time)
        
        txt_file = os.path.join(result_dir, image_file.split('/')[-1].split('.')[0]+'.txt')
        #print(txt_file)
        txt_f = open(txt_file, 'w')
        txt_f.write(extracted_text)
        # for key in result:
        #     print(result[key][1])
        #     txt_f.write(result[key][1]+'\n')
        txt_f.close()
    print("Total pipeline prediction time for all the images: ", total_e2e_time)
