import torchattacks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# from skimage.metrics import structural_similarity, peak_signal_noise_ratio

pretrained_model = "./model/lenet_mnist_model.pth"
# pretrained_model = "./model/ans_minist_nb_60.pth"
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



batch_size = 1 # 改进方法要求每个批次一定只能是1
# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True)

print(len(test_loader) * batch_size)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def show_cmp(ori_images, adv_images):
    '''
    显示原图像和对抗样本对比图
    '''

    # ori_images = ori_images.squeeze().detach().cpu().numpy()
    # adv_images = adv_images.squeeze().detach().cpu().numpy()

    # plt.subplot(1,2,1)
    # plt.title('Original image')
    # plt.imshow(ori_images, cmap="gray")
    # plt.subplot(1,2,2)
    # plt.title('Adversial image')
    # plt.imshow(adv_images, cmap="gray")
    # plt.show()

    ori_img1 = ori_images[0].detach().squeeze().cpu().numpy()
    ori_img2 = ori_images[1].detach().squeeze().cpu().numpy()
    adv_img1 = adv_images[0].detach().squeeze().cpu().numpy()
    adv_img2 = adv_images[1].detach().squeeze().cpu().numpy()

    plt.subplot(2, 2, 1)
    plt.title('Original image')
    plt.imshow(ori_img1, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title('Adversial image')
    plt.imshow(adv_img1, cmap="gray")
    plt.subplot(2, 2, 3)
    plt.imshow(ori_img2, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(adv_img2, cmap='gray')
    plt.show()

def similarity(ori_image, adv_image):
    """
    计算所有相似度
    """

    ori_image = ori_image.squeeze().detach().cpu().numpy()
    adv_image = adv_image.squeeze().detach().cpu().numpy()
    # print(ori_image.shape)
    # print(adv_image.shape)

    # 计算ssim(结构相似性度量)、PSNR、l2
    ssim = structural_similarity(ori_image, adv_image, data_range=255, multichannel=False)
    psnr = peak_signal_noise_ratio(ori_image, adv_image, data_range=255)
    l2 = np.linalg.norm(ori_image - adv_image)
    
    return ssim, psnr, l2

################ 开始攻击 ##################
epsilon = 0.3

attack = torchattacks.BIM_IMPROVE(model, eps=epsilon, alpha=1/255, iters=0)

correct = 0
ssim = 0
psnr = 0
l2 = 0
for data, target in test_loader:
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)
    # 防止对原始样本进行了更改
    data_t = data.clone()
    target_t = target.clone()
    flag, adv_images = attack(data_t, target_t)
    if flag == True:
        correct += 1

    # show_cmp(data, adv_images) # 显示图像，只能batch_size=1的时候才可以
    # break
    ssim_t, psnr_t, l2_t = similarity(data, adv_images)
    ssim += ssim_t
    psnr += psnr_t
    l2 += l2_t
    # print("SSIM: {}, PSNR: {} dB, L2: {}".format(ssim, psnr, l2))
    # continue

test_len = len(test_loader) * batch_size * 1.0
print("SSIM: {}, PSNR: {} dB, L2: {}".format(ssim/test_len, psnr/test_len, l2/test_len))

# Calculate final accuracy for this epsilon
final_acc = correct/test_len
print('Epsilon: {}\tAccuracy: {:.4f}'.format(epsilon, final_acc))