# -*- coding: utf-8 -*-

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import torchattacks
import numpy as np
import argparse
import logging
import torch
import time
import cv2
import os

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default="./data/pre_test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show original image and after attack image with the predicted result.')
    parser.add_argument('--pretrained_model', default='./weights/Ans_LPRNet_model_0.968.pth', help='pretrained base model')
    parser.add_argument('--epsilon', default=0.06, type=float, help='the degree of disturbance(0~1)')
    # args无法直接输入bool，所以默认设置成False，这样输入任何值都是True
    args = parser.parse_args()

    return args

def log():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = './Logs/' + rq + '.log' # 日志文件就存在当前路径下就好了
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
    # 定义logger的输出格式
    formatter = logging.Formatter("%(asctime)s - [line:%(lineno)d] %(message)s") 
    fh.setFormatter(formatter)
    logger.addHandler(fh) # 将logger添加到handler里面
    return logger

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

def test(args, logger):

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

    acc, ssim, psnr, l2 = Greedy_Decode_Eval(lprnet, test_dataset, args)
    logger.info("SSIM: {}, PSNR: {} dB, L2: {}".format(ssim, psnr, l2))
    logger.info("Epsilon: {}\tAccuracy: {:.4f}".format(args.epsilon, acc))
    print("Epsilon: {}\tAccuracy: {:.4f}".format(args.epsilon, acc))
    print("SSIM: {}, PSNR: {} dB, L2: {}".format(ssim, psnr, l2))

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size # 整除，多余的末尾就不会包括进来了
    # collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能 
    # shuffle：设置为True的时候，每个世代都会打乱数据集 
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
    attack = torchattacks.BIM_IMPROVE(Net, eps=args.epsilon, alpha=1/255, iters=0)

    # total = epoch_size * args.test_batch_size   # 总识别样本数
    total = epoch_size   # 为了保证准确率，批数为128同时进行识别，但只取每批次第一个样本作为代表。因此total=样本数/批数
    print(total)
    correct = 0
    ssim = 0
    psnr = 0
    l2 = 0
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)  # 提取iter的元素，注意images里面是整个batch的图像，但这时候类型是tensor了
        data = images.clone()
        # print(lengths) # 如果batch_size=100，那么lengths就是[7,7,...,7]，100个7(list类型)
        # print(images.shape)
        start = 0
        targets = []
        for length in lengths:  # 事到如今又要将tabel一个个提取出来，那为什么之前要用extend方法而不用append？
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets], dtype=np.int32)
        
        flag, perturbed_images = attack(images, labels, lengths, targets)
        if flag == True:
            correct += 1

        ssim_t, psnr_t, l2_t = similarity(data[0], perturbed_images[0]) # 本来也就只有一张图片
        ssim += ssim_t
        psnr += psnr_t
        l2 += l2_t

        print(i+1, correct)
    
    return (correct * 1.0 / total, ssim / total, psnr / total, l2 / total)


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

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def similarity(ori_image, adv_image):
    """
    计算所有相似度
    """

    # ori_image = ori_image.squeeze().detach().cpu().numpy()
    # adv_image = adv_image.squeeze().detach().cpu().numpy()
    ori_image = reduction(ori_image)
    adv_image = reduction(adv_image)
    # print(ori_image.shape)
    # print(adv_image.shape)


    # 计算ssim(结构相似性度量)、PSNR、l2
    ssim = structural_similarity(ori_image, adv_image, data_range=255, multichannel=True)
    psnr = peak_signal_noise_ratio(ori_image, adv_image, data_range=255)
    # l2 = np.linalg.norm(ori_image - adv_image)
    l2 = np.sqrt(np.sum((ori_image - adv_image)**2))
    
    return ssim, psnr, l2

def reduction(image):
    # 将归一化还原
    image = image.squeeze().detach().cpu().numpy()
    img = np.transpose(image, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    return img

if __name__ == "__main__":
    args = get_parser()
    logger = log()
    test(args, logger)