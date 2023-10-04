# Oct 16: 需要让y_inverse里z_2的和等于C

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import warnings
import math
warnings.filterwarnings("ignore", category=UserWarning)


class MLP(nn.Module):
    def __init__(self, num_outputs, num_hiddens):
        super().__init__()
        self.layer1 = nn.LazyLinear(num_hiddens)
        self.layer2 = nn.LazyLinear(num_outputs)

    def forward(self, x):
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

net = MLP(num_outputs=100, num_hiddens=256)
# print(model)
batch_size = 5
gt_data = torch.randn(batch_size, 1, 32, 32)
gt_onehot_label = torch.zeros((batch_size, 100))
gt_onehot_label[:, 1] += 1
criterion = l2_for_onehot
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
w_2_inv = torch.linalg.pinv(w_2)

grads = []
for name, params in net.named_parameters():
    grads.append(params.grad)

C = b_2 - gt_onehot_label - 0.5 * grads[3]
# I = torch.eye(8192)
res = torch.linalg.inv(w_2.T @ w_2) @ w_2.T @ (-C.T)
validation = w_2.T @ w_2 @ h.T + w_2.T @ C.T
validation_2 = w_2 @ h.T + C.T
my_pred = (1 / (w_2.T @ grads[3])) @ w_2.T @ grads[2] / 8192
print(1)
# res1, res2 = solve_power_2(weight_copy[2], weight_copy[3].unsqueeze(1) - gt_onehot_label.T, - grads[2])

# b_minus_y = gt_onehot_label - weight_copy[3].unsqueeze(1).T
# h_inverse = b_minus_y @ torch.linalg.pinv(weight_copy[2].T)

# I = torch.eye(8192)
# res = torch.kron(h.T, w_2) + torch.kron(I, w_2 @ h.T) + torch.kron(b_2.T, I) - torch.kron(gt_onehot_label.T, I)

# net_inv = Inverse_MLP(ori_layer1, ori_layer2)
# x_recover = net_inv(h)
# x_recover = net_inv(y_inverse)
# print("The error is {}.".format(torch.norm(x_recover - gt_data).item()))