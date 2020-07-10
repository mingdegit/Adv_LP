from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from PIL import Image, ImageDraw, ImageFont

os.chdir('/home/roxbili/Documents/BYSJ/test')

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "./model/lenet_mnist_model.pth"
use_cuda = True

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

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=False)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

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

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    adv_examples = []
    total = 0

    # Loop over all examples in test set
    for data, target in test_loader:

        if (total == 1):
            break
        total += 1

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item(): # 连一开始都预测不准的就不需要了
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Return the accuracy and an adversarial example
    return adv_examples[0][2]

def compute_all(epsilons, examples):
    """
    计算所有相似度
    """

    # 计算ssim(结构相似性度量)、PSNR、余弦相似度变化曲线
    ssim = []
    psnr = [100] # 由于0的时候相当于无穷
    l2 = []
    for i in range(len(epsilons)):
        ssim.append(structural_similarity(examples[0], examples[i], data_range=255, multichannel=False))
        if i > 0:
            psnr.append(peak_signal_noise_ratio(examples[0], examples[i], data_range=255))
        l2.append(np.linalg.norm(examples[0] - examples[i]))
    
    return ssim, psnr, l2





examples = []

# Run test for each epsilon
for eps in epsilons:
    ex = test(model, device, test_loader, eps)
    examples.append(ex)

ssim, psnr, l2 = compute_all(epsilons, examples)
print(epsilons)
print(ssim)
print(psnr)
print(l2)