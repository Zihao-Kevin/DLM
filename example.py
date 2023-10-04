import cv2
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import numpy as np

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.BatchNorm2d(12),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def prepare_dataloader(path=".", batch_size=64, shuffle=True):
    at_t_dataset_train = torchvision.datasets.CIFAR100(
        root=path, train=True, download=True
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = NumpyDataset(
        at_t_dataset_train.data,
        at_t_dataset_train.targets,
        transform=transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    return dataloader

torch.manual_seed(1)

shape_img = (28, 28)
num_classes = 10
channel = 1
hidden = 588

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
dataloader = prepare_dataloader()
for data in dataloader:
    x, y = data[0], data[1]
    break

criterion = nn.CrossEntropyLoss()
net = LeNet(channel=3, hideen=768, num_classes=100).to(device)
batch_size = 10


pred = net(x[:batch_size].to(device))
loss = criterion(pred, y[:batch_size].to(device))
received_gradients = torch.autograd.grad(loss, net.parameters())
received_gradients = [cg.detach() for cg in received_gradients]

dlg_attacker = GradientInversion_Attack(net, (3, 32, 32), lr=0.1, log_interval=0,
                                    num_iteration=1000, optimizer_class=torch.optim.Adam,
                                    optimize_label=False, tv_reg_coef=5,
                                    distancename="l2", device=device)

fig = plt.figure(figsize=(6, 3))
for i in range(batch_size):
  fig.add_subplot(1, batch_size, i+1)
  plt.imshow(cv2.cvtColor(x.detach().numpy()[i].astype(np.float32).transpose(1, 2, 0)*0.5+0.5, cv2.COLOR_BGR2RGB))
  plt.axis("off")
  plt.title(f"{y[i]}")
plt.tight_layout()
plt.savefig("true.png")
plt.show()

dlg_attacker.reset_seed(0)
result = dlg_attacker.attack(received_gradients, batch_size=batch_size)
fig = plt.figure(figsize=(6, 3))
for i in range(batch_size):
  fig.add_subplot(1, batch_size, i+1)
  plt.imshow(cv2.cvtColor(result[0].detach().cpu().numpy()[i].transpose(1, 2, 0)*0.5+0.5, cv2.COLOR_BGR2RGB))
  plt.title(f"{result[1][i]}")
  plt.axis("off")
plt.tight_layout()
plt.savefig("rec.png")
plt.show()