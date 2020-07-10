from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random

pretrained_model = "./model/lenet_mnist_model.pth"
# pretrained_model = "./model/ans_minist_1.pth"
use_cuda = True
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LoadMinist(Dataset):
    def __init__(self, img_dir):
        self.img_paths = []
        for root, dirs, files in os.walk(img_dir):
            for name in files:
                self.img_paths.append(os.path.join(root, name))

        random.shuffle(self.img_paths) # 改变图片顺序
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        filename = self.img_paths[index]
        # print(filename)
        data, target = torch.load(filename, map_location=device)
        return data, target

'''
# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
'''

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # print(image.shape) # torch.Size([1, 1, 28, 28]) [batch_size, 通道数， 图片尺寸, 图片尺寸]
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def train(model, device, train_dataset, epsilon=0.2, alpha=0.5):

    loss_val = 0
    cnt = 0
    for data, target in train_dataset:

        model.eval() # 先计算出原样本梯度和对抗样本梯度，然后相加

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        loss1 = F.nll_loss(output, target) # 先计算正确梯度
        loss1.backward(retain_graph=True) # 反向传播获得梯度信息
        data_grad = data.grad.data # 获得攻击方向
        perturbed_data = fgsm_attack(data, epsilon, data_grad) # 获得对抗样本
        output = model(perturbed_data) # 对抗样本输入网络前向传播
        loss2 = F.nll_loss(output, target) # 对抗样本的loss
        loss = alpha * loss1 + (1-alpha) * loss2 # 新的loss值
        model.train() # 打开模型训练模式

        optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha = 0.9, eps=1e-08,
                        momentum=0.9, weight_decay=2e-5)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val += loss.item()
        cnt += 1
        if (cnt % 30 == 0):
            loss_val /= 30
            print('Epoch: {}, Loss: {:.4f}'.format(i, loss_val))
            loss_val = 0

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 开始
# train_dataset = LoadMinist('./data/atk_img')
train_dataset = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=128, shuffle=True)

test_dataset = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=100, shuffle=True)

for i in range(30):
    train(model, device, train_dataset, epsilon=0.2)
    test(model, device, test_dataset)
    torch.save(model.state_dict(), './model/ans_minist.pth')


