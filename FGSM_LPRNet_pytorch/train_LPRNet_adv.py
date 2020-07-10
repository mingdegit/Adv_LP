# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import logging
import torch
import time
import os

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

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=60, type=int, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')

    parser.add_argument('--train_img_dirs', default="./data/pre_train", help='the train images path')
    parser.add_argument('--test_img_dirs', default="./data/pre_test", help='the test images path')
    parser.add_argument('--train_batch_size', default=128, type=int, help='training batch size.')
    parser.add_argument('--test_batch_size', default=100, type=int, help='testing batch size.')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    # parser.add_argument('--pretrained_model', default='', help='pretrained base model')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='base value of learning rate.') # 即使学习率为0，模型也会变成什么都识别不了
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')

    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

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

def train(args, logger, epsilon=0.04, alpha=0.5):

    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0
    cnt = 0
    # lr = args.learning_rate # 学习率

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print("load pretrained model successful!") # 从模型net.train()和net.eval()得出的结果完全不一样
        # test_img_dirs = os.path.expanduser(args.test_img_dirs) # 测试集
        # test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
        # Greedy_Decode_Eval(lprnet, test_dataset, args)
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    # os.path.expanduser把path中包含的"~"和"~user"转换成用户目录
    train_img_dirs = os.path.expanduser(args.train_img_dirs) # 训练集
    test_img_dirs = os.path.expanduser(args.test_img_dirs) # 测试集
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    epoch_size = len(train_dataset) // args.train_batch_size # 求得批数
    max_iter = args.max_epoch * epoch_size # 共需要循环的次数

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0: # 说明新的一个周期开始
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            cnt = 0 # 用于统计加了几次loss
            epoch += 1

        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth') # 经过一定的间隔后就保存状态

        if (iteration + 1) % args.test_interval == 0: # 经过一定间隔就评估模型
            Greedy_Decode_Eval(lprnet, test_dataset, args, logger)
            # lprnet.train() # should be switch to train mode

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.cuda:
            images = Variable(images.cuda(), requires_grad=True)
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=True)
            labels = Variable(labels, requires_grad=False)

        lprnet.eval() # 先开测试模式
        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()   # requires_grad_()相当于把requires_grad属性置为1
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)

        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # backprop
        loss1 = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        lprnet.zero_grad()
        loss1.backward(retain_graph=True)
        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)  # FGSM攻击

        logits = lprnet(perturbed_images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2).requires_grad_()   # requires_grad_()相当于把requires_grad属性置为1
        loss2 = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss = alpha * loss1 + (1-alpha) * loss2 # 新的loss值

        lprnet.train()
        lprnet.zero_grad()
        optimizer.zero_grad()   # 梯度置0
        loss.backward()
        optimizer.step()
        loss_val += loss.item() # 在输出的时候可以loss_val取平均
        cnt += 1 # loss_val的次数
        end_time = time.time()
        if (iteration + 1) % 20 == 0:
            msg = 'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) \
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f || ' % (loss_val / cnt) + \
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr)
            print(msg)
            logger.info(msg) # 存入日志
        # print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
        #           + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss_val / cnt) +
        #           'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))

    # final test
    print("Final test Accuracy:")
    logger.info("Final test Accuracy:")
    Greedy_Decode_Eval(lprnet, test_dataset, args, logger)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'Ans_LPRNet_model.pth')

def Greedy_Decode_Eval(Net, datasets, args, logger):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0 # 正确识别数
    Tn_1 = 0 # 字符缺失数
    Tn_2 = 0 # 字符未缺失但是识别错误数
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
    # 打印日志
    logger.info("Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    logger.info("Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))



if __name__ == "__main__":
    args = get_parser()
    logger = log()
    train(args, logger)
