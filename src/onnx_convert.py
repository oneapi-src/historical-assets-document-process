# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
import argparse
import torch
import torch.onnx
from torch.autograd import Variable
import crnn
import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument('-m',
                        '--fp32modelpath',
                        type=str,
                        required=True,
                        default="",
                        help='fp32 Model Path')
    parser.add_argument('-output',
                        '--onnxmodelpath',
                        type=str,
                        required=True,
                        default="",
                        help='onnx Model Path')

    FLAGS = parser.parse_args()
    config.model_path = FLAGS.fp32modelpath
    config.output_path = FLAGS.onnxmodelpath
    crnn = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh) # Initialize the mode
    crnn.load_state_dict(torch.load(config.model_path, map_location="cpu")) # Load the model
    model = crnn
    model.eval() # Set the mode to eval mode
    input_torch_tensor = Variable(torch.randn(1, 1, 32, 120)) # Dummy input initialization
    onnx_model_name = config.output_path+'/test_model.onnx' 
    torch.onnx.export(model, input_torch_tensor, onnx_model_name) # Save the model
