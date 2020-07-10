# -*- coding: utf-8 -*-
'''
计算不同epsilon下的准确率，并画出变化趋势
'''

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
import matplotlib.pyplot as plt
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default="./data/my_test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='use cuda to train model')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):  # batch中每个元素是 return Image, label, len(label)
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample # label是一个列表
        imgs.append(torch.from_numpy(img))
        labels.extend(label) # extend方法将label所有内容一次性加入labels中，若用append会把整个列表作为 一个 新的元素
        lengths.append(length)
    # 由于labels只是一个list，所以np.asarray(labels)就只是单纯复制一个相同的列表，变成numpy格式
    # np.flatten()返回一个折叠成一维的数组
    labels = np.asarray(labels).flatten().astype(np.float32)

    # 注意这里返回的是一个元组
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)    # toch.stack 增加新的维度进行堆叠，在0维度上堆叠就是把图片numpy数组组合起来，添加了一个新的维度

def sparse_tuple_for_ctc(T_length, lengths):
    '''
    制作CTC需要的元组
    '''
    input_lengths = [T_length for i in range(len(lengths))]
    target_lengths = lengths

    # 这是原先的代码，效率不高
    # input_lengths = []
    # target_lengths = []

    # for ch in lengths:
    #     input_lengths.append(T_length)
    #     target_lengths.append(ch)
    # print(input_lengths)
    # print(target_lengths)
    return tuple(input_lengths), tuple(target_lengths)

def test():
    args = get_parser()
    # epsilons = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04] # 0.07时能降到10%左右
    epsilons = np.arange(0, 0.1, 0.005).tolist()
    accuracies = []

    # 返回Net.train()或Net.eval()
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)   # 实例化后使用.to方法将网络移动到GPU或CPU
    print("Successful to build network!")   # 到此位置模型搭建完成

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print("load pretrained model successful!\n")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)  # 把path中包含的"~"和"~user"转换成用户目录
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len) # lpr_max_len为车牌最大字符数
    epoch_size = len(test_dataset) // args.test_batch_size # 整除，多余的末尾就不会包括进来了
    test_dataset = DataLoader(test_dataset, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    # collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能 
    # shuffle：设置为True的时候，每个世代都会打乱数据集 

    examples = []
    for epsilon in epsilons:
        batch_iterator = iter(test_dataset)
        perturbed_image = Greedy_Decode_Eval(lprnet, batch_iterator, args, epsilon, epoch_size)
        b, g, r = cv2.split(perturbed_image)
        perturbed_image = cv2.merge([r, g, b])
        examples.append(perturbed_image)
    return compute_all(epsilons, examples) # 画图

def compute_all(epsilons, examples):
    """
    计算所有相似度
    """

    # 计算ssim(结构相似性度量)、PSNR、余弦相似度变化曲线
    ssim = []
    psnr = [100] # 由于0的时候相当于无穷
    l2 = []
    for i in range(len(epsilons)):
        ssim.append(structural_similarity(examples[0], examples[i], data_range=255, multichannel=True))
        if i > 0:
            psnr.append(peak_signal_noise_ratio(examples[0], examples[i], data_range=255))
        l2.append(np.sqrt(np.sum((examples[i] - examples[0])**2)))
    
    return ssim, psnr, l2

def Greedy_Decode_Eval(Net, batch_iterator, args, epsilon, epoch_size):
    # TestNet = Net.eval()
    # t1 = time.time()
    total = epoch_size * args.test_batch_size   # 总识别样本数
    atk_reg_correct = 0 # 攻击后还识别正确

    images, labels, lengths = next(batch_iterator)  # 提取iter的元素，注意images里面是整个batch的图像，但这时候类型是tensor了
    start = 0
    targets = []
    for length in lengths:  # 事到如今又要将tabel一个个提取出来，那为什么之前要用extend方法而不用append？
        label = labels[start:start+length]
        targets.append(label)
        start += length
    targets = np.array([el.numpy() for el in targets])
    imgs = images.numpy().copy()    # 转成numpy格式的副本，copy方法和原对象不共享内存，是完完全全的赋值一个新的

    if args.cuda:   # 这里Variable不知道为什么，删除后无法正常运行代码
        images = Variable(images.cuda(), requires_grad = True)  # requires_grad = True由后面的requires_grad_()函数来改
        labels = Variable(labels, requires_grad=False).cuda() # 后面计算Loss需要把labels变成tensor
    else:
        images = Variable(images, requires_grad = True)
        labels = Variable(labels, requires_grad=False)

    # orinin images forward
    prebs = Net(images) # prebs是个tensor
    # print(images.shape) # [batch_size, 3, 24, 94] 3个通道，24x94是图片尺寸
    # print(images.grad)

    '''
    FGSM attack
    '''
    # epsilons = [0, .05, .1, .15, .2, .25, .3]
    # epsilon = args.epsilon
    log_probs = prebs.permute(2, 0, 1) # for ctc loss: T x N x C, llog_probs.shape = (18, 100, 68)
    
    # requires_grad_()相当于把requires_grad属性置为1;softmax的作用简单的说就计算一组数值中每个值的占比
    log_probs = log_probs.log_softmax(2).requires_grad_()
    T_length = 18
    input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)

    # 计算loss
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
    Net.zero_grad() # 先清空已经存在的梯度
    loss.backward()
    
    data_grad = images.grad.data    # 这个必须要进行反向传播后才有值
    perturbed_images = fgsm_attack(images, epsilon, data_grad)  # FGSM攻击

    perturbed_img = perturbed_images[0].detach().cpu().numpy() # 这句话提取了攻击后的图像
    # 将归一化还原
    perturbed_img = np.transpose(perturbed_img, (1, 2, 0))
    perturbed_img *= 128.
    perturbed_img += 127.5
    perturbed_img = perturbed_img.astype(np.uint8)

    print(epsilon)
    return perturbed_img

def get_preb_labels(prebs):
    '''
    通过识别的结果获得最后预测的车牌标签
    '''
    prebs = prebs.cpu().detach().numpy()    # .detach()会返回requires_grab = False的版本，但是注意和prebs共享存储空间。此时prebs是numpy数组
    preb_labels = list()
    for i in range(prebs.shape[0]): # shape[0]为最高维度数，prebs.shape = (100, 68, 18)，100是每批有100张图片矩阵
        preb = prebs[i, :, :]   # preb.shape = (68, 18)，注意这个68正是CHARS的字符数，表达68类，要在每个字符里在68类找到概率最大的才行
        preb_label = list()
        '''
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0)) # np.argmax 取出a中元素最大值所对应的索引
        '''
        preb_label = np.argmax(preb, axis=0).tolist()    # 按列方向找，和上面的一个效果，没必要每一列那样遍历，最后得到18个元素的列表

        # 这段代码目的是得到车牌正确的字符索引，但是感觉很奇怪，如果数字全是一样的车牌怎么办？preb_label为18个元素，是车牌字符的两倍以上，字符和字符中间会识别出空白，所以重复也没顾上你洗
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        # print(preb_label)
        if pre_c != len(CHARS) - 1: # 67索引字符是'-'，代表空白，即不存在这个字符
            no_repeat_blank_label.append(pre_c)
        for c in preb_label: # dropout repeate label and blank label（除去的是相邻重复字符和空白字符）
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label) # 将每次的预测存入列表
    return preb_labels

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    '''
    攻击函数（注意：由于每张图片扰动方向都不一致，按批处理效果不会比单张图片好，但是省时间，具体自己把握）
    '''
    # print(images.shape) # [batch_size, 3, 24, 94] 3个通道，24x94是图片尺寸; torch.tensor类型
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()   # 我们只为了得知扰动的方向，具体扰动的值由epsilon来确定。因此用sign来提取斜率的符号即可
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # Adding clipping to maintain [-1,1] range（将每个元素夹紧到-1~1的区间内）
    '''
        | min, if x_i < min
    y_i = | x_i, if min <= x_i <= max
        | max, if x_i > max
    '''
    perturbed_image = torch.clamp(perturbed_image, -1, 1)

    # Return the perturbed image
    return perturbed_image

if __name__ == "__main__":
    ssim, psnr, l2 = test()
    epsilons = np.arange(0, 0.1, 0.005).tolist()
    print(epsilons)
    print(ssim)
    print(psnr)
    print(l2)