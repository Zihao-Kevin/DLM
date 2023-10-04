# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from defense import *

from utils import label_to_onehot, cross_entropy_for_onehot
import lpips

parser = argparse.ArgumentParser(description='Deep Leakage from Model.')
parser.add_argument('--index', type=int, default="60",
                    help='the index for leaking images on dataset.')
parser.add_argument('--dataset', type=str,default="cifar100",
                    help='the name of data set')
parser.add_argument('--network_name', type=str,default="MLP",
                    help='the name of network')
parser.add_argument('--defense_type', type=str,default="None",
                    help='the defense type')
parser.add_argument('--local_step', type=bool,default=False,
                    help='start local step experiment or not')
args = parser.parse_args()


global count
count = 0

# defense_type dp spars None
defense_type = args.defense_type

net_name = args.network_name

def DLM_plus(index, seed, defense_strenth=0.00001, dptype='laplace', local_step=1):
    torch.manual_seed(seed)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)
    global count

    dataset=args.dataset
    if dataset=="tinyimagenet":
        from TinyImageNetLoader import TinyImageNet
        dst=TinyImageNet("./tiny-imagenet-200", train=True)
        class_num=200
    elif dataset == "cifar100":
        dst = datasets.CIFAR100("~/.torch", download=True)
        class_num=100
    else:
        print("unsurport")
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()

    img_index = index
    gt_data = tp(dst[img_index][0]).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)

    input_dim = int(torch.prod(torch.tensor(gt_data.size())))

    opt_name=""
    maxit=0
    from models.vision import MLP2, LeNet, weights_init
    if net_name=="MLP":
        net = MLP2(input_dim, class_num).to(device)
        net.apply(weights_init)
        net1 = MLP2(input_dim, class_num).to(device)
        net1.apply(weights_init)
        net1.load_state_dict(net.state_dict())
        opt_name="adam"
        maxit=5000
    elif net_name == "lenet":
        net = LeNet(input_dim, class_num).to(device)
        net.apply(weights_init)
        net1 = LeNet(input_dim, class_num).to(device)
        net1.apply(weights_init)
        net1.load_state_dict(net.state_dict())
        # opt_name="adam"
        # maxit=10000
        opt_name="lbfgs"
        maxit=200

    criterion = cross_entropy_for_onehot

    optimizer_ori = torch.optim.SGD(net.parameters(), lr=0.1)


    if defense_type == "dp":
        last_layer1 = dp_defense(net, dp_strength=defense_strenth, type=dptype ,device=device)
    elif defense_type == "spars":
        last_layer1 = sparsification(net, spars=defense_strenth, device=device)
    else:
        last_layer1 = copy.deepcopy(net.state_dict())

    net1.load_state_dict(last_layer1)

    for i in range(local_step):
        pred = net(gt_data)
        loss = criterion(pred, gt_onehot_label)
        loss.backward()
        optimizer_ori.step()

    if defense_type == "dp":
        last_layer2 = dp_defense(net, dp_strength=defense_strenth, type=dptype ,device=device)
    elif defense_type == "spars":
        last_layer2 = sparsification(net, spars=defense_strenth, device=device)
    else:
        last_layer2 = copy.deepcopy(net.state_dict())
    original_dy_dx = []
    ori_norm = 0
    for param, k in zip(last_layer1, last_layer2):
        original_dy_dx.append(last_layer1[param] - last_layer2[param])
    ori_norm = torch.stack([g.norm() for g in original_dy_dx]).mean()

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = (torch.randn(gt_onehot_label.size())).to(device).requires_grad_(True)

    optimizer=0
    if opt_name=="adam":
        optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = 1)


    print("ori norm = {}".format(ori_norm))

    dis_lpips = lpips.LPIPS(net='alex')

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_history = []
    history = []
    max_it=200
    psnr_history=[]
    for iters in range(max_it):

        def closure():
            optimizer.zero_grad()

            dummy_pred = net1(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)

            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net1.parameters(), create_graph=True)

            dummy_norm = torch.stack([g.norm() for g in dummy_dy_dx]).mean()
            grad_diff = 0

            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx / dummy_norm - gy / ori_norm) ** 2).sum()
            grad_diff.backward()

            return grad_diff


        optimizer.step(closure)
        current_loss = closure()
        loss_history.append(current_loss.item())

        mse = torch.mean((dummy_data - gt_data) ** 2).item()
        cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
        ds = torch.as_tensor(cifar100_std)[:, None, None]
        factor = 1 / ds
        psnrs = 10 * torch.log10(factor ** 2 / mse)
        psnr = torch.mean(psnrs)
        psnr_history.append(psnr.item())

        if iters % 50 == 0:
            current_loss = closure()
            print("----------------------------------------------------")
            print("Iters = ", iters, "Loss = ", current_loss.item())
            history.append(tt(dummy_data[0].cpu()))

            mse = torch.mean((dummy_data - gt_data) ** 2).item()
            cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
            ds = torch.as_tensor(cifar100_std)[:, None, None]
            factor = 1 / ds
            psnrs = 10 * torch.log10(factor ** 2 / mse)
            psnr = torch.mean(psnrs)

            ss_dummy_data = copy.deepcopy(dummy_data).to("cpu")
            ss_gt_data = copy.deepcopy(gt_data).to("cpu")
            LPIPS = dis_lpips(ss_dummy_data, ss_gt_data).item()
            print("PSNR = ", psnr.item())
            print("LPIPS = ", LPIPS)
            print("----------------------------------------------------")


    return loss_history,psnr_history

import math
def ExpName(param):
    if param ==0:
        return 0
    else:
        return math.log10(param)

img_index = args.index

if args.defense_type == "dp":
    experiments_list = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    dptype = ["laplace", "gaussian"]
    round = 5

    for r in range(round):
        seed = np.random.randint(0, 2 ** 32 / 2 - 100)
        for type in dptype:
            for i in experiments_list:
                loss_history, psnr_history = DLM_plus(img_index, seed, i, type)
                file = open("dp_{}_loss_{}_{}_({}).txt".format(net_name, type, ExpName(i), r), 'w')
                file.write(str(loss_history))
                file.close()

                file = open("dp_{}_psnr_{}_{}_({}).txt".format(net_name, type, ExpName(i), r), 'w')
                file.write(str(psnr_history))
                file.close()

if args.defense_type == "spars":
    for r in range(round):
        seed = np.random.randint(0, 2 ** 32 / 2 - 100)
        for i in range(10):
            loss_history, psnr_history = DLM_plus(img_index, seed, i)
            file = open("spars_{}_loss_{}_({}).txt".format(net_name, i * 10, r), 'w')
            file.write(str(loss_history))
            file.close()

            file = open("spars_{}_psnr_{}_({}).txt".format(net_name, i * 10, r), 'w')
            file.write(str(psnr_history))
            file.close()

if args.local_step == True:
    local_step = [1]
    local_step.extend([i for i in range(20, 101, 20)])
    for r in range(round):
        for i in local_step:
            loss_history, psnr_history = DLM_plus(img_index, local_step=i)

            file = open("{}_loss_step_{}({}).txt".format(net_name, i, r), 'w')
            file.write(str(loss_history))
            file.close()

            file = open("{}_psnr_step_{}({}).txt".format(net_name, i, r), 'w')
            file.write(str(psnr_history))
            file.close()