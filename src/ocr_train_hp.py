# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-docstring

from __future__ import print_function
import argparse
import random
import os
import datetime
import time
import torch
from torch import optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn import CTCLoss
import numpy as np
import utils  # pylint : disable=no-name-in-module
import mydataset
import crnn
import config
from online_test import val_model

def weights_init(m):
    '''This function initialises the weights for the model if there are no pretrained weights
    input:model'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def val(net, dataset, criterion, max_iter=100):
    '''val function Validates the model to determine the accuracy and saves the best model
       input params :model,dataset,criterion
       output params : accuracy
    '''
    best_acc = 0.1
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False
    #Validate the model
    num_correct, num_all = val_model(config.val_infofile, net, False, log_file='compare-'+config.saved_model_prefix+'.log')
    #print(num_correct)
    #print(num_all)
    accuracy = num_correct / num_all # Calculate accuracy
    print('ocr_acc: %f' % (accuracy))
    if config.use_log:
        with open(log_filename, 'a', encoding="utf-8") as f:
            f.write('ocr_acc:{}\n'.format(accuracy))
    if accuracy > best_acc: # Check if the accuracy got is greater than the best accuracy
        best_acc = accuracy # Set best accuracy as the latest accuracy
        torch.save(net.state_dict(), '{}/{}_{}_{}.pth'.format(config.saved_model_dir, config.saved_model_prefix, "best",
                                                              int(best_acc * 1000))) #Save the model
    return accuracy

def trainBatch(net, criterion, optimizer):
    '''trainBatch function trains the crnn model and returns the cost
       input params:net,criterion,optimizer
       output param:cost '''
    data = train_iter.next() # Get next batch of data 
    cpu_images, cpu_texts = data # get the images and labels
    batch_size = cpu_images.size(0)
    print("batch size: " + str(batch_size))
    image = cpu_images
    text, length = converter.encode(cpu_texts)
    preds = net(image)  # seqLength x batchSize x alphabet_size
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size).cpu())  # seqLength x batchSize
    cost = criterion(preds.log_softmax(2).cpu(), text.cpu(), preds_size, length.cpu()) / batch_size # Compute Cost
    if torch.isnan(cost): # Check if the cost is nan value
        print(batch_size, cpu_texts)
    else:
        net.zero_grad()
        cost.backward() #Backward propagation
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step() # optimizer is applied
    return cost

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--model_prefix',
                        type=str,
                        required=True,
                        default="",
                        help='Prefix for saved model')
    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 for enabling intel pytorch optimizations, default is 0')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=8,
                        help='batch size for training')
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        default="",
                        help='Pre-Trained Model Path')

    FLAGS = parser.parse_args()
    intel_flag = FLAGS.intel
    config.batchSize = FLAGS.batch_size
    config.pretrained_model = FLAGS.model_path
    config.saved_model_prefix = FLAGS.model_prefix
    
    # Initialize the config values
    torch.autograd.set_detect_anomaly(True)
    
    config.alphabet = config.alphabet_v2
    config.nclass = len(config.alphabet) + 1

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    log_filename = os.path.join('logs/', 'loss_acc-'+config.saved_model_prefix + '.log')
    if not os.path.exists('debug_files'):
        os.mkdir('debug_files')
    if not os.path.exists(config.saved_model_dir):
        os.mkdir(config.saved_model_dir)
    if config.use_log and not os.path.exists('logs'):
        os.mkdir('logs')
    if config.use_log and os.path.exists(log_filename):
        os.remove(log_filename)
    if config.experiment is None:
        config.experiment = 'expr'
    if not os.path.exists(config.experiment):
        os.mkdir(config.experiment)

    config.manualSeed = random.randint(1, 10000)  # nosec # fix seed
    random.seed(config.manualSeed)
    np.random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    train_dataset = mydataset.MyDataset(info_filename=config.train_infofile)
    assert train_dataset  # nosec
    if not config.random_sample:
        sampler = mydataset.randomSequentialSampler(train_dataset, config.batchSize) # Random sampling of data
    else:
        sampler = None
    # Load train data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batchSize,
                                               shuffle=True, sampler=sampler, num_workers=int(config.workers),
                                               collate_fn=mydataset.alignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio=config.keep_ratio))
    # Load test data
    test_dataset = mydataset.MyDataset(info_filename=config.val_infofile,
                                       transform=mydataset.resizeNormalize((config.imgW, config.imgH), is_test=True))
    converter = utils.strLabelConverter(config.alphabet)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    # custom weights initialization called on crnn

    if config.cuda:
        crnn.cuda()
        device = torch.device('cuda:0')
        criterion = criterion.cuda()

    crnn = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh) # Initialize the model
    if config.pretrained_model != '' and os.path.exists(config.pretrained_model):
        print('loading pretrained model from %s' % config.pretrained_model)
        crnn.load_state_dict(torch.load(config.pretrained_model, map_location="cpu")) # Load the pretrained model if it exists
    else:
        crnn.apply(weights_init) # else apply initial weights
    print(crnn)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if config.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    elif config.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=config.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=config.lr)

    if intel_flag: # Chek if intel flag is enabled
        import intel_extension_for_pytorch as ipex  # pylint: disable=E0401
        crnn = ipex.optimize(crnn, optimizer=optimizer) # Optimize the model using ipex
        crnn_net = crnn[0] # assign the crnn model to variable
        optimizer_intel = crnn[1] # assign the optimizer to variable
        print("Intel Pytorch Optimizations has been Enabled!")
    else:
        device = torch.device('cpu')

    parameters_list = []
    accuracy_list = []
    print("Start of hp tuning ")
    train_time = time.time()
    batch_size = [80] # initialize the batch size 
    epochs_vals = [5, 10] # initialize the epochs values
    lr_vals = [1e-3] # initialize the learning rate
    for bt_sz in batch_size:  # iterating through the batch_size list
        config.batchSize = bt_sz
        for epoch_val in epochs_vals:  # iterating through epoch values list
            config.niter = epoch_val
            for lrng_rate in lr_vals:  # iterating through the learning rate list
                config.lr = lrng_rate

                for epoch in range(config.niter):
                    loss_avg.reset()
                    print('epoch {}....'.format(epoch))
                    train_iter = iter(train_loader)
                    i = 0
                    n_batch = len(train_loader)
                    if intel_flag:
                        while i < len(train_loader):
                            for p in crnn_net.parameters():
                                p.requires_grad = True
                            crnn_net.train()
                            cost = trainBatch(crnn_net, criterion, optimizer_intel)
                            print('epoch: {} iter: {}/{} Train loss: {:.3f}'.format(epoch, i, n_batch, cost.item()))
                            loss_avg.add(cost)
                            #loss_avg.add(cost)
                            i += 1
                        print('Train loss: %f' % (loss_avg.val()))
                        if config.use_log:
                            with open(log_filename, 'a', encoding="utf-8") as f:
                                f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
                                f.write('train_loss:{}\n'.format(loss_avg.val()))
                    else:
                        while i < len(train_loader):
                            for p in crnn.parameters():
                                p.requires_grad = True
                            crnn.train()
                            cost = trainBatch(crnn, criterion, optimizer)
                            print('epoch: {} iter: {}/{} Train loss: {:.3f}'.format(epoch, i, n_batch, cost.item()))
                            loss_avg.add(cost)
                            #loss_avg.add(cost)
                            i += 1
                        print('Train loss: %f' % (loss_avg.val()))
                        if config.use_log:
                            with open(log_filename, 'a', encoding="utf-8") as f:
                                f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
                                f.write('train_loss:{}\n'.format(loss_avg.val()))
                    #val(crnn, test_dataset, criterion)
                print("Inferencing.................")
                if intel_flag:
                    valid_accuracy = val(crnn_net, test_dataset, criterion) # Validate accuracy for intel 
                else:
                    valid_accuracy = val(crnn, test_dataset, criterion) # Validate accuracy for stock
                accuracy_list.append(valid_accuracy)
                parameters_list.append((config.batchSize, config.lr, config.niter))
    print("Hyperparameter tuning time is", time.time()-train_time)
    max_value = max(accuracy_list)
    max_index = accuracy_list.index(max_value)
    print("accuracy list")
    print(accuracy_list)
    print("parameters list")
    print(parameters_list)
    print("the best parameters are")
    print(parameters_list[max_index])
