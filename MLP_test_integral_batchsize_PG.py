# Oct 16: 需要让y_inverse里z_2的和等于C

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import warnings
import math
warnings.filterwarnings("ignore", category=UserWarning)


class MLP(nn.Module):
    def __init__(self, input_dim, num_outputs, num_hiddens):
        super().__init__()
        self.layer0 = nn.Flatten()
        self.layer1 = nn.Linear(int(input_dim / batch_size), num_hiddens)
        self.layer2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):
        x = self.layer0(x)
        z = self.layer1(x)
        h = F.sigmoid(z)
        h = torch.flatten(h, 1)     # 5, 8192
        z_2 = self.layer2(h)
        return z, h, z_2


def Softmax_Inverse(y_hat, repeat_times):
    C = 10
    z_2 = None
    for _ in range(repeat_times):
        z_2 = torch.log(y_hat) + math.log(C)
        C = torch.sum(torch.exp(z_2))
    return z_2


def Sigmoid_Inverse(x):
    x = - torch.log(1 / x - 1)
    return x


class Linear_Inverse(nn.Module):
    def __init__(self, weight, bias, **kwargs):
        super(Linear_Inverse, self).__init__(**kwargs)
        self.ori_weight = weight
        self.ori_bias = bias

    def forward(self, x):
        x = (x - self.ori_bias) @ torch.linalg.pinv(self.ori_weight).T
        return x


class Inverse_MLP(nn.Module):
    def __init__(self, ori_layer_1, ori_layer_2):
        super(Inverse_MLP, self).__init__()
        self.layer1 = Linear_Inverse(ori_layer_2[0], ori_layer_2[1])
        self.layer2 = Linear_Inverse(ori_layer_1[0], ori_layer_1[1])

    def forward(self, x):
        # x = z^(2)
        x = self.layer1(x)
        # x = h.flatten
        x = x.reshape(batch_size, 1, 32, 256)
        # x = h
        x = Sigmoid_Inverse(x)
        # x = z^(1)
        x = self.layer2(x)
        # x = X and return
        return x


def cross_entropy_for_onehot(pred, target):
    y_hat = F.softmax(pred, dim=-1)
    return y_hat, torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def l2_for_onehot(pred, target):
    return pred, torch.norm(pred - target) ** 2


def solve_power_2(a, b, c):
    return (- b + torch.sqrt(b ** 2 - 4 * a * c)) / 2 * a, (- b - torch.sqrt(b ** 2 - 4 * a * c)) / 2 * a


batch_size = 10
gt_data = torch.randn(batch_size, 3, 32, 32)
input_dim = int(torch.prod(torch.tensor(gt_data.size())))

gt_onehot_label = torch.zeros((batch_size, 100))
gt_onehot_label[:, 1] += 1
y = gt_onehot_label

net = MLP(input_dim=input_dim, num_outputs=100, num_hiddens=512)
# print(model)

criterion = cross_entropy_for_onehot
z, h, z_2, y_hat = None, None, None, None
for i in range(1):
    z, h, z_2 = net(gt_data)
    y_hat, loss = criterion(z_2, gt_onehot_label)
    loss.backward()
    # optimizer_ori.step()

# y_inverse = Softmax_Inverse(y_hat, repeat_times=1)

weight_copy = []
for param in net.state_dict():
    tmp = copy.deepcopy(net.state_dict()[param])
    weight_copy.append(tmp)

w_1 = weight_copy[0]
b_1 = weight_copy[1]
w_2 = weight_copy[2]
b_2 = weight_copy[3]
grads = []
for name, params in net.named_parameters():
    grads.append(params.grad)

# my_pred = (1 / (w_2.T @ grads[3].reshape(100, 1))) @ w_2.T @ grads[2] / grads[2].shape[1]
# validation = w_2 @ h.T @ ones.T + b_2.view(-1, 1) @ ones @ ones.T - y.T @ ones.T - 0.5 * grads[3].view(100, 1)
# pinv_h = torch.linalg.pinv(h)

# grads[2] @ torch.linalg.pinv(h) @ ones.T == grads[3].view(-1, 1)

ones = torch.ones(batch_size).view(1, -1)
Eye = torch.eye(batch_size)
Lambda = 1
beta = 1.25
max_iter = 5000
Delta_b = grads[3].view(-1, 1)

X_k = torch.ones_like(h)

for iter in range(max_iter):
    updated = torch.linalg.pinv(ones.T @ Delta_b.T @ Delta_b @ ones + Lambda * Eye) \
             @ (Lambda * X_k + ones.T @ Delta_b.T @ grads[2])
    Lambda *= beta
    X_k = copy.deepcopy(updated)
    print(torch.norm(X_k - h).item())