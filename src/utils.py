# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=consider-using-from-import
# pylint disable=R1705
#!/usr/bin/python
# encoding: utf-8

import collections
from datetime import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable

def get_acc(output, label):
    """get_acc finds the accuracy from the predicted output and the groundtruth label
    input params :output and groundtruth
    output params : accuracy """

    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    # print( pred_label.data.cpu().numpy() )
    # print( label.data.cpu().numpy() )
    return 1.0*num_correct / total

def adjust_learning_rate(optimizer, decay_rate=0.97):
    """adjust_learning_rate adjusts the learning rate
    input params :ooptimizer and decay rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay_rate

def train(net, train_data, valid_data, num_epochs, optimizer, criterion, saver_freq=50, saver_prefix='vgg16'):
    """train function trains the model and saved the trained model 
    input params :net, train_data, valid_data, num_epochs, optimizer saver_freq,criterion,prefix)"""    

    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    best_acc = 0.98
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            # print(label)
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda(), volatile=True)
                    label = Variable(label.cuda(), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    label = Variable(label, volatile=True)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
            if valid_acc / len(valid_data) > best_acc:
                best_acc = valid_acc / len(valid_data)
                torch.save(net.state_dict(), 'models/{}-{}-{}-0819-model-db.pth'.format(saver_prefix, epoch + 1, int(best_acc*1000)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        #print(epoch_str + time_str)
        adjust_learning_rate(optimizer)

class strLabelConverter():
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
       # init method or constructor
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet
        self.alphabet.append(ord('_'))  # for `-1` index
        # print(self.alphabet)

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        # print(text)
        try:
            if isinstance(text, str):
                # for char in text:
                #     print(char)
                text = [
                    self.dict[ord(char.lower() if self._ignore_case else char)]
                    for char in text  # if char in self.dict.keys()
                ]
                length = [len(text)]
            elif isinstance(text, collections.abc.Iterable):
                length = [len(s) for s in text]
                text = ''.join(text)
                text, _ = self.encode(text)
        except KeyError as e:
            print(e)
            for ch in text:
                if ord(ch) not in self.dict.keys():
                    print('Not Covering Char: {} - {}'.format(ch, ord(ch)))
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)  # nosec
            if raw:  # pylint: disable=R1705
                return ''.join([chr(self.alphabet[i - 1]) for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(chr(self.alphabet[t[i] - 1]))

                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())  # nosec
            texts = []
            index = 0
            for i in range(length.numel()):
                str_len = length[i]
                texts.append(
                    self.decode(
                        t[index:index + str_len], torch.IntTensor([str_len]), raw=raw))
                index += str_len
            return texts


class averager(object):  # pylint: disable=useless-object-inheritance
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        # init method or constructor
        self.reset()

    def add(self, v):
        """This function adds all the values"""
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        """This function computes the average values"""
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    """This function Perform oneHot encoding
    input params :vector input
    """
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


