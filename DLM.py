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

parser = argparse.ArgumentParser(description='Deep Leakage from Model.')
parser.add_argument('--dataset', type=str,default="cifar100",
                    help='the name of data set')
parser.add_argument('--network_name', type=str,default="MLP",
                    help='the name of network')
args = parser.parse_args()



net_name = args.network_name


def DLM(index, k_init = 200):
    # torch.manual_seed(2022)
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

    gt_data = gt_data.view(1, *gt_data.size()).to(device)
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

    optimizer_ori = torch.optim.SGD(net.parameters(), lr=0.01)

    last_layer1 = []
    for param in net.state_dict():
        tmp = copy.deepcopy(net.state_dict()[param])
        last_layer1.append(tmp)

    # compute original gradient
    for i in range(1):
        pred = net(gt_data)
        loss = criterion(pred, gt_onehot_label)
        loss.backward()
        optimizer_ori.step()

    original_dy_dx = []
    ori_norm = 0
    for param, k in zip(net.state_dict(), last_layer1):
        tmp = copy.deepcopy(net.state_dict()[param])
        original_dy_dx.append(k - tmp)
        # ori_norm += torch.linalg.norm(k - tmp)
    ori_norm = torch.stack([g.norm() for g in original_dy_dx]).mean()

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = (torch.randn(gt_onehot_label.size())).to(device).requires_grad_(True)

    k = torch.tensor(k_init).requires_grad_(True)
    optimizer=0
    if opt_name=="adam":
        optimizer = torch.optim.Adam([dummy_data, dummy_label, k], lr=0.1)
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label, k], lr = 1)


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
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net1.parameters(), create_graph=True)

            dummy_norm = torch.stack([g.norm() for g in dummy_dy_dx]).mean()

            grad_diff = 0

            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - k * gy) ** 2).sum()


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

            LPIPS = dis_lpips(dummy_data, gt_data).item()
            ssim_val = ssim(dummy_data, gt_data, data_range=255, size_average=True).item()
            print("PSNR = ", psnr.item())
            print("LPIPS = ", LPIPS)
            print("SSIM = ", ssim_val)
            print("----------------------------------------------------")

    return loss_history,psnr_history


trials = 1
repeat_times = 10
index_list = [i for i in range(60, 60 + trials)]

k_init_list=[60.,80.,100.,120.,140.,160.,180.,200.,220.,240.]
# k_init_list=[100.]
for image_index in index_list:
    for k in k_init_list:
        for i in range(repeat_times):
            loss_history, psnr_history = DLM(image_index, k)
            file = open("{}_loss_k_{}_image_{}({}).txt".format(net_name, k, image_index, i), 'w')
            file.write(str(loss_history))
            file.close()

            file = open("{}_psnr_k_{}_image_{}({}).txt".format(net_name, k, image_index, i), 'w')
            file.write(str(psnr_history))
            file.close()