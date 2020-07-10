# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

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
    parser.add_argument('--test_img_dirs', default="./data/my_test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
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

    # 返回Net.train()或Net.eval()
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)   # 实例化后使用.to方法将网络移动到GPU或CPU
    print("Successful to build network!")   # 到此位置模型搭建完成

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)  # 把path中包含的"~"和"~user"转换成用户目录
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len) # lpr_max_len为车牌最大字符数
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    # collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能 
    # shuffle：设置为True的时候，每个世代都会打乱数据集 
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
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
            images = Variable(images.cuda(), requires_grad=False)  # requires_grad = True由后面的requires_grad_()函数来改
            labels = Variable(labels, requires_grad=False).cuda() # 后面计算Loss需要把labels变成tensor
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # orinin images forward
        prebs = Net(images) # prebs是个tensor
        # print(images.grad)

        """
        '''
        查看loss
        '''
        log_probs = prebs.permute(2, 0, 1) # for ctc loss: T x N x C, llog_probs.shape = (18, 100, 68)
        
        # requires_grad_()相当于把requires_grad属性置为1;softmax的作用简单的说就计算一组数值中每个值的占比
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.requires_grad_()
        T_length = 18
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)

        # 计算loss
        ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        print(loss.item())
        """

        # greedy decode
        prebs = prebs.cpu().detach().numpy()    # .detach()会返回requires_grab = False的版本，但是注意和prebs共享存储空间。此时prebs是numpy数组
        # print(prebs)
        preb_labels = list()
        for i in range(prebs.shape[0]): # shape[0]为最高维度数，prebs.shape = (100, 68, 18)，100是每批有100张图片矩阵
            preb = prebs[i, :, :]   # preb.shape = (68, 18)，注意这个68正是CHARS的字符数，表达68类，要在每个字符里在68类找到概率最大的才行
            preb_label = list()
            '''
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0)) # np.argmax 取出a中元素最大值所对应的索引
            '''
            preb_label = np.argmax(preb, axis=0).tolist()    # 按列方向找，和上面的一个效果，没必要每一列那样遍历，最后得到18个元素的列表

            # 这段代码目的是得到车牌正确的字符索引，但是感觉很奇怪，如果数字全是一样的车牌怎么办？preb_label为18个元素，是车牌字符的两倍以上，字符和字符中间会识别出空白，所以重复也没关系
            # 想要读懂这段代码需要对ctc_loss机制有个大致的了解，关键在于blank部分和整体字符的关系上是怎么处理的
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

        for i, label in enumerate(preb_labels):
            # show image and its predict label
            if args.show:
                show(imgs[i], label, targets[i])
            if len(label) != len(targets[i]): # targets每个元素还是一个列表，每个元素就是一个车牌号的字符索引值
                Tn_1 += 1   # 字符缺失数
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all(): # 判断是否每一个元素都相等
                Tp += 1 # 正确识别数
            else:
                Tn_2 += 1 # 字符未缺失但识别错误数
    # print(Tp, Tn_1, Tn_2)
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2) # 计算正确率
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {} s/per_img]".format((t2 - t1) / len(datasets)))

def show(img, label, target): # 显示图像，可选项
    img = np.transpose(img, (1, 2, 0))
    # 将归一化还原
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
    # img = cv2ImgAddText(img, lb, (0, 0))
    # cv2.imshow("test", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    test()
