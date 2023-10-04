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

from utils import label_to_onehot, cross_entropy_for_onehot

import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from skimage.metrics import peak_signal_noise_ratio
dis_lpips = lpips.LPIPS(net='alex')

parser = argparse.ArgumentParser(description='Deep Leakage from Model.')
parser.add_argument('--dataset', type=str,default="cifar100",
                    help='the name of data set')
parser.add_argument('--network_name', type=str,default="MLP",
                    help='the name of network')
args = parser.parse_args()

net_name = args.network_name


def DLG(seed):
    torch.manual_seed(seed)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)

    dataset = args.dataset
    if dataset == "tinyimagenet":
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

    img_index = args.index
    gt_data = tp(dst[img_index][0]).to(device)

    if len(args.image) > 1:
        gt_data = Image.open(args.image)
        gt_data = tp(gt_data).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=class_num)
    input_dim = int(torch.prod(torch.tensor(gt_data.size())))
    plt.imshow(tt(gt_data[0].cpu()))


    opt_name = ""
    maxit = 0
    from models.vision import MLP2, LeNet, weights_init

    if net_name == "MLP":
        net = MLP2(input_dim, class_num).to(device)
        net.apply(weights_init)
        opt_name = "adam"
        maxit = 10000
    elif net_name == "lenet":
        net = LeNet(input_dim, class_num).to(device)
        net.apply(weights_init)
        # opt_name = "adam"
        # maxit = 10000
        opt_name = "lbfgs"
        maxit = 300

    # torch.manual_seed(1234)

    criterion = cross_entropy_for_onehot

    # compute original gradient
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    plt.imshow(tt(dummy_data[0].cpu()))
    if opt_name == "adam":
        optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=1)

    history = []
    max_it = maxit
    loss_history = []
    psnr_history = []
    lpips_history = []
    ssim_history = []
    for iters in range(max_it):
        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
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

        ss_dummy_data = copy.deepcopy(dummy_data).to("cpu")
        ss_gt_data = copy.deepcopy(gt_data).to("cpu")
        LPIPS = dis_lpips(ss_dummy_data, ss_gt_data).item()
        ssim_val = ssim(dummy_data, gt_data, data_range=255, size_average=True).item()
        lpips_history.append(LPIPS)
        ssim_history.append(ssim_val)

        if iters % 10 == 0:
            current_loss = closure()
            print("----------------------------------------------------")
            print("Iters = ", iters, "Loss = ", current_loss.item())
            # history.append(tt(dummy_data[0].cpu()))

            mse = torch.mean((dummy_data - gt_data) ** 2).item()
            cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
            ds = torch.as_tensor(cifar100_std)[:, None, None]
            factor = 1 / ds
            psnrs = 10 * torch.log10(factor ** 2 / mse)
            psnr = torch.mean(psnrs)

            print("PSNR = ", psnr.item())
            print("----------------------------------------------------")

        # if iters % int(max_it / 10) == 0:
        #     history.append(tt(dummy_data[0].cpu()))

    return loss_history,psnr_history, lpips_history, ssim_history


def DLG_k(seed, k_init):
    torch.manual_seed(seed)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)

    dataset = "cifar100"
    if dataset == "tinyimagenet":
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

    img_index = args.index
    gt_data = tp(dst[img_index][0]).to(device)

    if len(args.image) > 1:
        gt_data = Image.open(args.image)
        gt_data = tp(gt_data).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=class_num)
    input_dim = int(torch.prod(torch.tensor(gt_data.size())))
    plt.imshow(tt(gt_data[0].cpu()))


    opt_name = ""
    maxit = 0
    from models.vision import MLP2, LeNet, weights_init

    if net_name == "MLP":
        net = MLP2(input_dim, class_num).to(device)
        net.apply(weights_init)
        opt_name = "adam"
        maxit = 10000
    elif net_name == "lenet":
        net = LeNet(input_dim, class_num).to(device)
        net.apply(weights_init)
        # opt_name = "adam"
        # maxit = 10000
        opt_name = "lbfgs"
        maxit = 300



    criterion = cross_entropy_for_onehot

    # compute original gradient
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    k = torch.tensor(k_init).to(device).requires_grad_(True)

    if opt_name == "adam":
        optimizer = torch.optim.Adam([dummy_data, dummy_label, k], lr=0.1)
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label, k], lr=1)

    history = []
    max_it = maxit
    loss_history = []
    psnr_history = []
    lpips_history = []
    ssim_history = []
    for iters in range(max_it):
        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx*k - gy) ** 2).sum()
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

        ss_dummy_data = copy.deepcopy(dummy_data).to("cpu")
        ss_gt_data = copy.deepcopy(gt_data).to("cpu")
        LPIPS = dis_lpips(ss_dummy_data, ss_gt_data).item()
        ssim_val = ssim(dummy_data, gt_data, data_range=255, size_average=True).item()
        lpips_history.append(LPIPS)
        ssim_history.append(ssim_val)
        if iters % 10 == 0:
            current_loss = closure()
            print("----------------------------------------------------")
            print("Iters = ", iters, "Loss = ", current_loss.item())
            # history.append(tt(dummy_data[0].cpu()))

            mse = torch.mean((dummy_data - gt_data) ** 2).item()
            cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
            ds = torch.as_tensor(cifar100_std)[:, None, None]
            factor = 1 / ds
            psnrs = 10 * torch.log10(factor ** 2 / mse)
            psnr = torch.mean(psnrs)
            print("PSNR = ", psnr.item())
            print("----------------------------------------------------")

        # if iters % int(max_it / 10) == 0:
        #     history.append(tt(dummy_data[0].cpu()))

    return loss_history, psnr_history, lpips_history, ssim_history

repeat_times=100
seed_history=[]
k_list=[0.9, 0.95, 1.05, 1.1]

# import ast
# seed_file1 = open("seed_history.txt")
# strlist = seed_file1.read()
# seed_list1 = ast.literal_eval(strlist)
# seed_file1.close()
#
# seed_file2 = open("seed_history_2.txt")
# strlist = seed_file2.read()
# seed_list2 = ast.literal_eval(strlist)
# seed_file2.close()
#
# seed_list = seed_list1
# seed_list.extend(seed_list2)

i=0
for a in range(repeat_times):
    seed =np.random.randint(0, 2**32 /2 - 100)
    # seed = seed_list[i]
    i += 1
    loss_history, psnr_history, lpips_history, ssim_history = DLG(seed)
    file = open("dlg_{}_loss_{}.txt".format(net_name,a), 'w')
    file.write(str(loss_history))
    file.close()
    file = open("dlg_{}_psnr_{}.txt".format(net_name,a), 'w')
    file.write(str(psnr_history))
    file.close()
    file = open("dlg_{}_lpips_{}.txt".format(net_name,a), 'w')
    file.write(str(lpips_history))
    file.close()
    file = open("dlg_{}_ssim_{}.txt".format(net_name,a), 'w')
    file.write(str(ssim_history))
    file.close()

    for k_init in k_list:
        loss_history, psnr_history, lpips_history, ssim_history = DLG_k(seed, k_init)
        file = open("dlgk_{}_psnr_{}_{}.txt".format(net_name,a, k_init), 'w')
        file.write(str(psnr_history))
        file.close()
        file = open("dlgk_{}_loss_{}_{}.txt".format(net_name,a, k_init), 'w')
        file.write(str(loss_history))
        file.close()
        file = open("dlgk_{}_lpips_{}_{}.txt".format(net_name,a, k_init), 'w')
        file.write(str(lpips_history))
        file.close()
        file = open("dlgk_{}_ssim_{}_{}.txt".format(net_name,a, k_init), 'w')
        file.write(str(ssim_history))
        file.close()

    seed_history.append(seed)

print("Seed history:{}".format(seed_history))
file = open("seed_history.txt", 'w')
file.write(str(seed_history))
file.close()
