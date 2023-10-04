# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

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

global count
count = 0

def DLG(img_index):
    torch.manual_seed(124)
    global count
    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda"
    print("Running on %s" % device)

    dst = datasets.CIFAR100("~/.torch", download=True)
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()

    gt_data = tp(dst[img_index][0]).to(device)
    #
    # if len(args.image) > 1:
    #     gt_data = Image.open(args.image)
    #     gt_data = tp(gt_data).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    # gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = torch.tensor(25).long().to(device)

    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)

    plt.imshow(tt(gt_data[0].cpu()))

    from models.vision import LeNet, weights_init, weights_init_Resnet, Resnet18
    net = LeNet().to(device)
    net.apply(weights_init)

    criterion = cross_entropy_for_onehot

    # compute original gradient
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    # plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    # optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
    dis_lpips = lpips.LPIPS(net='alex')


    history = []
    for iters in range(200):
        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            costs = 0
            pnorm = [0, 0]
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                costs -= (gx * gy).sum()
                pnorm[0] += gx.pow(2).sum()
                pnorm[1] += gy.pow(2).sum()

            grad_diff = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
            # for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            #     grad_diff += 1 - torch.nn.functional.cosine_similarity(gx.flatten(),
            #                                                        gy.flatten(),
            #                                                        0, 1e-10)
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
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

            LPIPS = dis_lpips(dummy_data, gt_data).item()
            ssim_val = ssim(dummy_data, gt_data, data_range=255, size_average=True).item()
            print("PSNR = ", psnr.item())
            print("LPIPS = ", LPIPS)
            print("SSIM = ", ssim_val)
            print("----------------------------------------------------")
    if psnr >= 25:
        count += 1
    plt.figure(figsize=(12, 5))
    for i in range(len(history)):
        plt.subplot(4, 5, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')

    plt.show()
    return psnr, LPIPS, ssim_val
trials = 50
index_list = [i for i in range(60, trials + 60)]
PSRN = []
LPIPS = []
SSIM_VAL = []
for i in index_list:
    a, b, c = DLG(i)
    if a >= 25:
        PSRN.append(a)
        LPIPS.append(b)
        SSIM_VAL.append(c)
print("Test Acc = ", count / trials)
print("PNSR AVG = ", np.sum(PSRN) / trials)
print("LPIPS AVG = ", np.sum(LPIPS) / trials)
print("SSIM AVG = ", np.sum(SSIM_VAL) / trials)