# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path')

args = parser.parse_args()
dataset_path = args.dataset_path

trainfile = open(dataset_path+"/train.txt", "w+")
testfile = open(dataset_path+"/test.txt", "w+")
images_path = dataset_path+"/dataset/"

images = os.listdir(images_path)
count = 0
for i in range(len(images)):
    i1 = images[i].index("_")
    s = dataset_path+"/dataset/"
    s+=images[i]+'\t'+images[i].split(".")[0][i1+1:len(images[i])]
    if count<100:
        testfile.write(s)
        if i!=(len(images)-1):
            testfile.write("\n")
        count+=1
    else:
        trainfile.write(s)
        if i!=(len(images)-1):
            trainfile.write("\n")
        count+=1

trainfile.close()
testfile.close()


