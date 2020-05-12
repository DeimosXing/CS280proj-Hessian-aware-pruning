#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function

import json
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from density_plot import get_esd_plot
from models.resnet import resnet
from pyhessian import hessian
from models import resnet_LT
from archs.mnist import fc1

# Settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument(
    '--mini-hessian-batch-size',
    type=int,
    default=200,
    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residual connect or not')

parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')
parser.add_argument('--zipname',
                    type=str,
                    default='mnist.zip')
parser.add_argument('--dataset',
                    type=str,
                    default='mnist')
parser.add_argument('--output',
                    type=str,
                    default='output')
parser.add_argument('--model',
                    type=str,
                    default='fc1')
parser.add_argument('--compute_each_layer',
                    action='store_true')
parser.add_argument('--extract',
                    action='store_false')
parser.add_argument('--parallel',
                    action='store_true')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

if args.dataset == "cifar10":
# get dataset
    train_loader, test_loader = getData(name='cifar10_without_dataaugmentation',
                                        train_bs=args.mini_hessian_batch_size,
                                        test_bs=1)

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
if args.dataset == "mnist":
    traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
    testdataset = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.mini_hessian_batch_size, shuffle=True, num_workers=0,drop_last=False)
    #train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.mini_hessian_batch_size, shuffle=False, num_workers=0,drop_last=True)

##############
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
assert (50000 % args.hessian_batch_size == 0)
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

if batch_num == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break
else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == batch_num - 1:
            break

# get model
# model = resnet(num_classes=10,
#                depth=20,
#                residual_not=args.residual,
#                batch_norm_not=args.batch_norm)
if args.model == 'resnet':
    model = resnet_LT.resnet18()
if args.model == 'fc1':
    model = fc1.fc1()

if args.cuda:
    model = model.cuda()
if args.parallel:
    model = torch.nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()  # label loss

###################
# Get model checkpoint, get saving folder
###################
if args.resume == '':
    raise Exception("please choose the trained model")

import zipfile, os
if args.extract:
    zipfile.ZipFile(args.resume + '/' + args.zipname).extractall(args.resume)
# print(args.resume + '/' + args.zipname)
# print(args.resume)
files = os.listdir(args.resume)

# for file_index in range(2):
for file_index in range(len(files)):
    if files[file_index][-4:]!='.pth':
        continue
    if not (files[file_index][0]>='0' and files[file_index][0]<='9'):
        index = '000_'
        if files[file_index][-6]=='h':
        	index += '00'
        elif not(files[file_index][-6]>='0' and files[file_index][-6]<='9'):
        	index += 'best'
        else:
        	index += files[file_index][-6:-4]
    elif not (files[file_index][1]>='0' and files[file_index][1]<='9'):
        index = int(files[file_index][0]) + 1
        if index == 10:
            index = '010_'
        else:
            index = '00' + str(index) +'_'
        if files[file_index][-6]=='h':
            index += '00'
        elif not(files[file_index][-6]>='0' and files[file_index][-6]<='9'):
            index += 'best'
        else:
            index += files[file_index][-6:-4]
    else:
        index = int(files[file_index][0:2]) + 1
        if index < 100:
            index = '0' + str(index) + '_'
        if files[file_index][-6]=='h':
        	index += '00'
        elif not(files[file_index][-6]>='0' and files[file_index][-6]<='9'):
            index += 'best'
        else:
        	index += files[file_index][-6:-4]
    print("\n*** This is " + files[file_index] + ", its index is " + index)
    # model.load_state_dict(torch.load(args.resume + '/' + files[file_index], map_location = torch.device('cpu')))
    
    # state_dict = torch.load('checkpoint.pt')
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v

    model.load_state_dict(torch.load(args.resume + '/' + files[file_index]))

    ######################################################
    # Begin the computation
    ######################################################

    # turn model to eval mode
    model.eval()

    if batch_num == 1:
        hessian_comp = hessian(model,
                               criterion,
                               data=hessian_dataloader,
                               cuda=args.cuda)
    else:
        hessian_comp = hessian(model,
                               criterion,
                               dataloader=hessian_dataloader,
                               cuda=args.cuda)

    print(
        '********** finish data londing and begin Hessian computation **********')
    print('now is for the whole model')

    top_eigenvalues, _ = hessian_comp.eigenvalues()
    trace = hessian_comp.trace()
    density_eigen, density_weight = hessian_comp.density()

    if not os.path.exists(args.output + "/top_eigenvalues"):
       os.makedirs(args.output + "/top_eigenvalues")
    f_top_eigenvalues = open(args.output+ '/top_eigenvalues' + '/top_eigenvalues' + index + '.txt', 'w')
    f_top_eigenvalues.write(str(top_eigenvalues[0]))
    f_top_eigenvalues.close()

    if not os.path.exists(args.output + "/trace"):
       os.makedirs(args.output + "/trace")
    f_trace = open(args.output + '/trace' + '/trace' + index + '.txt', 'w')
    f_trace.write(str(np.mean(trace)))
    f_trace.close()

    print('\n***Top Eigenvalues: ', top_eigenvalues)
    print('\n***Trace: ', np.mean(trace))


    # print(density_eigen)
    # print(density_weight)
    if not os.path.exists(args.output + "/density_eigen"):
       os.makedirs(args.output + "/density_eigen")
    f_density_eigen = open(args.output + '/density_eigen' + '/density_eigen' + index + '.txt', 'w')
    for i in range(len(density_eigen[0])):
        f_density_eigen.write(str(density_eigen[0][i]) + '\n')
    f_density_eigen.close()

    if not os.path.exists(args.output + "/density_weight"):
       os.makedirs(args.output + "/density_weight")
    f_density_weight = open(args.output + '/density_weight' + '/density_weight' + index + '.txt', 'w')
    for i in range(len(density_weight[0])):
        f_density_weight.write(str(density_weight[0][i]) + '\n')
    f_density_weight.close()

    if not os.path.exists(args.output + "/esd_plot"):
       os.makedirs(args.output + "/esd_plot")
    get_esd_plot(density_eigen, density_weight, path = args.output + '/esd_plot' + '/esd_plot' + index + '.pdf')
    print("\n***finish " + index +"\n\n")


    if args.compute_each_layer:
        layer = -1
        for param in model.parameters():
            layer += 1
            if not param.requires_grad:
                continue
            if batch_num == 1:
                hessian_comp = hessian(model,
                                       criterion,
                                       data=hessian_dataloader,
                                       cuda=args.cuda,
                                       whole_model=False,
                                       layer=layer)
            else:
                hessian_comp = hessian(model,
                                       criterion,
                                       dataloader=hessian_dataloader,
                                       cuda=args.cuda,
                                       whole_model=False,
                                       layer=layer)

            print(
                '********** finish data londing and begin Hessian computation **********')
            print('now is for layer ' + str(layer))

            top_eigenvalues, _ = hessian_comp.eigenvalues()
            trace = hessian_comp.trace()
            density_eigen, density_weight = hessian_comp.density()

            if not os.path.exists(args.output + "/top_eigenvalues_layer" + str(layer)):
               os.makedirs(args.output + "/top_eigenvalues_layer" + str(layer))
            f_top_eigenvalues = open(args.output+ '/top_eigenvalues_layer' + str(layer) + '/top_eigenvalues' + index + '.txt', 'w')
            f_top_eigenvalues.write(str(top_eigenvalues[0]))
            f_top_eigenvalues.close()

            if not os.path.exists(args.output + "/trace_layer" + str(layer)):
               os.makedirs(args.output + "/trace_layer" + str(layer))
            f_trace = open(args.output + '/trace_layer' + str(layer) + '/trace' + index + '.txt', 'w')
            f_trace.write(str(np.mean(trace)))
            f_trace.close()

            print('\n***Top Eigenvalues: ', top_eigenvalues)
            print('\n***Trace: ', np.mean(trace))


            if not os.path.exists(args.output + "/density_eigen_layer" + str(layer)):
               os.makedirs(args.output + "/density_eigen_layer" + str(layer))
            f_density_eigen = open(args.output + '/density_eigen_layer' + str(layer) + '/density_eigen' + index + '.txt', 'w')
            for i in range(len(density_eigen[0])):
                f_density_eigen.write(str(density_eigen[0][i]) + '\n')
            f_density_eigen.close()

            if not os.path.exists(args.output + "/density_weight_layer" + str(layer)):
               os.makedirs(args.output + "/density_weight_layer" + str(layer))
            f_density_weight = open(args.output + '/density_weight_layer' + str(layer) + '/density_weight' + index + '.txt', 'w')
            for i in range(len(density_weight[0])):
                f_density_weight.write(str(density_weight[0][i]) + '\n')
            f_density_weight.close()

            if not os.path.exists(args.output + "/esd_plot_layer" + str(layer)):
               os.makedirs(args.output + "/esd_plot_layer" + str(layer))
            get_esd_plot(density_eigen, density_weight, path = args.output + '/esd_plot_layer' + str(layer) + '/esd_plot' + index + '.pdf')
            print("\n***finish " + index +"\n\n")
