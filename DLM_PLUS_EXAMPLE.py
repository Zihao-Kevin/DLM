# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
# import matplotlib
# matplotlib.use('Agg')
from PIL import Image
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=Warning)
from defense import *
# import models.tinyimagenet_resnet as tinynet

from utils import label_to_onehot, cross_entropy_for_onehot
import lpips

parser = argparse.ArgumentParser(description='Deep Leakage from Model.')
parser.add_argument('--dataset', type=str, default="tiny_imagenet",
                    help='the name of data set')
parser.add_argument('--network_name', type=str, default="mlp",
                    help='the name of network')
args = parser.parse_args()
net_name = args.network_name

def DLM_plus(index, batch_size):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)
    global count

    dataset = args.dataset
    if dataset == "tiny_imagenet":
        from TinyImageNetLoader import TinyImageNet
        dst = TinyImageNet("./tiny-imagenet-200", train=True)
        class_num = 200
    elif dataset == "cifar100":
        dst = datasets.CIFAR100("~/.torch", download=True)
        class_num = 100
    else:
        print("unsurport")

    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()

    # gt_data = tp(dst[img_index][0]).to(device)
    gt_data = []
    gt_label = []
    for i in index:
        gt_data.append(tp(dst[i][0]).to(device))
        label = torch.Tensor([dst[i][1]]).long().to(device)
        gt_onehot_label = label_to_onehot(label.view(1, ), num_classes=class_num)
        gt_label.append(gt_onehot_label)
    gt_data = torch.stack(gt_data).to(device)
    gt_label = torch.stack(gt_label).to(device)

    plt.figure(figsize=(12, 12))
    if batch_size == 1:
        plt.subplot(1, 2, 1)
        plt.imshow(tt(gt_data[0].cpu()))
        plt.title("iter=%d" % (0 * 10))
        plt.axis('off')
    else:
        for i in range(batch_size):
            plt.subplot(int(batch_size / 2), 2, i + 1)
            plt.imshow(tt(gt_data[i].cpu()))
            plt.title("iter=%d" % (i * 10))
            plt.axis('off')

    input_dim = int(torch.prod(torch.tensor(gt_data.size())))

    opt_name = ""
    maxit = 0
    from models.vision import MLP2, LeNet, weights_init
    if net_name == "mlp":
        net = MLP2(input_dim, class_num, batch_size=batch_size).to(device)
        net.apply(weights_init)
        net1 = MLP2(input_dim, class_num, batch_size=batch_size).to(device)
        net1.apply(weights_init)
        net1.load_state_dict(net.state_dict())
        opt_name = "lbfgs"
        maxit = 50
        print_it = 10
        # opt_name = "adam"
        # maxit = 2000
        # print_it = 200
    elif net_name == "lenet":
        net = LeNet(input_dim, class_num, batch_size=batch_size).to(device)
        net.apply(weights_init)
        net1 = LeNet(input_dim, class_num, batch_size=batch_size).to(device)
        net1.apply(weights_init)
        net1.load_state_dict(net.state_dict())
        # opt_name="adam"
        # maxit=10000
        opt_name = "lbfgs"
        maxit = 50
        print_it = 10

    criterion = cross_entropy_for_onehot
    optimizer_ori = torch.optim.SGD(net.parameters(), lr=0.1)

    last_layer1 = []
    for param in net.state_dict():
        tmp = copy.deepcopy(net.state_dict()[param])
        last_layer1.append(tmp)

    # local update times
    for i in range(1):
        pred = net(gt_data)
        loss = criterion(pred, gt_onehot_label)
        loss.backward()
        optimizer_ori.step()

    original_dy_dx = []
    ori_norm = 0
    for param, k in zip(net.state_dict(), last_layer1):
        if 'weight' not in param and 'bias' not in param or 'mask' in param:
            continue
        tmp = copy.deepcopy(net.state_dict()[param])
        original_dy_dx.append(k - tmp)
    ori_norm = torch.stack([g.norm() for g in original_dy_dx]).mean()

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = (torch.randn(gt_onehot_label.size())).to(device).requires_grad_(True)

    # optimizer=0
    if opt_name == "adam":
        optimizer = torch.optim.Adam([dummy_data], lr=0.1)
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS([dummy_data], lr=1)

    print("ori norm = {}".format(ori_norm))

    dis_lpips = lpips.LPIPS(net='alex')

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_history = []
    metric_history = []
    history = []
    loss_gb = 0
    iterss = 0
    max_it=maxit
    psnr_history=[]
    for iters in range(max_it):

        def closure():
            optimizer.zero_grad()

            dummy_pred = net1(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            # dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(dummy_pred, -1)), dim=-1))
            dummy_loss = criterion(dummy_pred, gt_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net1.parameters(), create_graph=True)

            dummy_norm = torch.stack([g.norm() for g in dummy_dy_dx]).mean()

            grad_diff = 0

            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += (1 * (gx / dummy_norm - gy / ori_norm) ** 2).sum()

            grad_diff.backward()

            return grad_diff


        optimizer.step(closure)
        current_loss = closure()
        # loss_history.append(current_loss.item())

        mse = torch.mean((dummy_data - gt_data) ** 2).item()
        cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
        ds = torch.as_tensor(cifar100_std)[:, None, None]
        factor = 1 / ds
        psnrs = 10 * torch.log10(factor ** 2 / mse)
        psnr = torch.mean(psnrs)
        # psnr_history.append(psnr.item())

        # if iters % int(max_it/10) == 0:
        #     history.append(tt(dummy_data[0].cpu()))
        if iters % print_it == 0:
            current_loss = closure()
            print("----------------------------------------------------")
            print("Iters = ", iters, "Loss = ", current_loss.item())

            mse = torch.mean((dummy_data - gt_data) ** 2).item()
            cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
            ds = torch.as_tensor(cifar100_std)[:, None, None]
            factor = 1 / ds
            psnrs = 10 * torch.log10(factor ** 2 / mse)
            psnr = torch.mean(psnrs)

            ss_dummy_data = copy.deepcopy(dummy_data).to('cpu')
            ss_gt_data = copy.deepcopy(gt_data).to('cpu')
            # LPIPS = dis_lpips(ss_dummy_data, ss_gt_data).item()
            print("PSNR = ", psnr.item())
            # print("LPIPS = ", LPIPS)
            print("----------------------------------------------------")
    # if psnr >= 20:
    #     count += 1

    plt.figure(figsize=(12, 5))
    if batch_size == 1:
        plt.subplot(1, 2, 1)
        plt.imshow(tt(dummy_data[0].cpu()))
        plt.title("iter=%d" % (max_it))
        plt.axis('off')
    else:
        for i in range(batch_size):
            plt.subplot(int(batch_size / 2), 2, i + 1)
            plt.imshow(tt(dummy_data[i].cpu()))
            plt.title("iter=%d" % (i * 10))
            plt.axis('off')
    plt.show()
    return loss_history, psnr_history

trials = 1
repeat_times = 1
# index_list = [i for i in range(60, 60 + trials)]
index_list = [90, 91]
batch_size = 2

for i in range(repeat_times):
    loss_history, psnr_history = DLM_plus(index_list, len(index_list))
    # print(loss_history[-1], psnr_history[-1])