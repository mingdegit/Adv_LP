# -*- coding: utf-8 -*-

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
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
    parser.add_argument('--test_img_dirs', default="./data/pre_train", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=100, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show original image and after attack image with the predicted result.')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--epsilon', default=0.04, type=float, help='the degree of disturbance(0~1)')

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
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()
    print('Done')

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size # 整除，多余的末尾就不会包括进来了
    # collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能 
    # shuffle：设置为True的时候，每个世代都会打乱数据集 
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    t1 = time.time()
    reg_wrong = 0   # 识别错误数
    total = epoch_size * args.test_batch_size   # 总识别样本数
    atk_reg_correct = 0 # 攻击后还识别正确

    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)  # 提取iter的元素，注意images里面是整个batch的图像，但这时候类型是tensor了
        # print(images.shape)
        start = 0
        targets = []
        for length in lengths:  # 事到如今又要将tabel一个个提取出来，那为什么之前要用extend方法而不用append？
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()    # 转成numpy格式的副本，copy方法和原对象不共享内存，是完完全全的赋值一个新的

        if args.cuda:   # 这里Variable不知道为什么，删除后无法正常运行代码
            images = Variable(images.cuda(), requires_grad=True)  # requires_grad = True由后面的requires_grad_()函数来改
            labels = Variable(labels, requires_grad=False).cuda() # 后面计算Loss需要把labels变成tensor
        else:
            images = Variable(images, requires_grad=True)
            labels = Variable(labels, requires_grad=False)

        # orinin images forward
        prebs = Net(images) # prebs是个tensor
        # print(images.grad)

        '''
        FGSM attack
        '''
        # epsilons = [0, .05, .1, .15, .2, .25, .3]
        epsilon = args.epsilon
        log_probs = prebs.permute(2, 0, 1) # for ctc loss: T x N x C, llog_probs.shape = (18, 100, 68)
        
        # requires_grad_()相当于把requires_grad属性置为1;softmax的作用简单的说就计算一组数值中每个值的占比
        log_probs = log_probs.log_softmax(2).requires_grad_()
        T_length = 18
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)

        # 计算loss
        ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        # print(loss.item())
        Net.zero_grad() # 先清空已经存在的梯度
        loss.backward()
        
        data_grad = images.grad.data    # 这个必须要进行反向传播后才有值
        perturbed_images = fgsm_attack(images, epsilon, data_grad)  # FGSM攻击

        for i in range(len(targets)):
            perturbed_img = perturbed_images[i].detach().cpu().numpy() # 这句话提取了攻击后的图像
            # 将归一化还原
            perturbed_img = np.transpose(perturbed_img, (1, 2, 0))
            perturbed_img *= 128.
            perturbed_img += 127.5
            perturbed_img = perturbed_img.astype(np.uint8)
            
            filename = ''
            for j in targets[i]:
                filename += CHARS[int(j)]
            filename += '.jpg'
            filepath = os.path.join('./data/train', filename)
            cv2.imwrite(filepath, perturbed_img)
        

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
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
    test()
