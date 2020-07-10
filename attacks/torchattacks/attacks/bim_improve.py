import torch
import torch.nn as nn
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
import numpy as np

from ..attack import Attack

class BIM_IMPROVE(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 4/255)
        alpha (float): step size. (DEFALUT : 1/255)
        iters (int): max iterations. (DEFALUT : 0)
    
    .. note:: If iters set to 0, iters will be automatically decided following the paper.
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.BIM(model, eps=4/255, alpha=1/255, iters=0)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=4/255, alpha=1/255, iters=0):
        super(BIM_IMPROVE, self).__init__("BIM_IMPROVE", model)
        self.eps = eps
        self.alpha = alpha
        if iters == 0 :
            self.iters = int(min(eps*255 + 4, 1.25*eps*255))
        else :
            self.iters = iters
        
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

    def forward(self, images, labels, lengths, targets):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels_ori = targets[0]
        T_length = 18
        input_lengths, target_lengths = BIM_IMPROVE.sparse_tuple_for_ctc(T_length, lengths)
        ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

        for i in range(self.iters) :    
            images.requires_grad = True
            outputs = self.model(images)
            log_probs = outputs.permute(2, 0, 1) # for ctc loss: T x N x C, llog_probs.shape = (18, 100, 68)
            log_probs = log_probs.log_softmax(2).requires_grad_()
            cost = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]
            
            adv_images = images + self.alpha*grad.sign()
            
            a = torch.clamp(images - self.eps, min=-1, max=1)
            b = (adv_images>=a).float()*adv_images + (a>adv_images).float()*a
            c = (b > images+self.eps).float()*(images+self.eps) + (images+self.eps >= b).float()*b
            images = torch.clamp(c, max=1).detach()

            # 测试当前阶段生成的对抗样本是否正确
            preb_atk = self.model(images)
            preb_labels_atk = np.array(BIM_IMPROVE.get_preb_labels(preb_atk))

            # print(preb_labels_atk[0])
            # print(labels_ori)
            # print(i)

            if len(preb_labels_atk[0]) != len(labels_ori) or (preb_labels_atk[0] != labels_ori).all():
                return (False, images)

        print("True")
        return (True, images)
    