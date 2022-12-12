# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring
import keys
train_infofile = 'data/train.txt'
train_infofile_fullimg = ''
val_infofile = 'data/test.txt'
alphabet = keys.alphabet
alphabet_v2 = keys.alphabet_v2
workers = 2
batchSize = 50
imgH = 32
imgW = 128
nc = 1
nclass = len(alphabet)+1
nh = 256
niter = 25
lr = 1e-3
beta1 = 0.5
cuda = False
ngpu = 0
pretrained_model = ''
saved_model_dir = 'src/crnn_models'
saved_model_prefix = 'CRNN-'
use_log = True
remove_blank = False
experiment = None
displayInterval = 500
n_test_disp = 10
valInterval = 500
saveInterval = 500
adam = True
adadelta = False
keep_ratio = True
random_sample = True
