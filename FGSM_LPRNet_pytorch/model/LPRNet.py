import torch.nn as nn
import torch
# import torch.nn.functional as F

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),   # //为整除
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        '''
        in_channels：输入维度
        out_channels：输出维度
        kernel_size：卷积核大小
        stride：步长大小(每次卷积核的窗口滑动的距离数)
        padding：补0
        dilation：kernel间距
        '''
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),    # 批标准化
            nn.ReLU(),  # 2     # reLu是激活函数
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),  # 类似于3维卷积，只不过不做卷积操作，只取当前窗口最大值作为新图的像素值。
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            # print(i, x.shape)   # 看看每一层的输出到底是什么
            '''
            0 torch.Size([1, 64, 22, 92])
            1 torch.Size([1, 64, 22, 92])
            2 torch.Size([1, 64, 22, 92])
            3 torch.Size([1, 64, 20, 90])
            4 torch.Size([1, 128, 20, 90])
            5 torch.Size([1, 128, 20, 90])
            6 torch.Size([1, 128, 20, 90])
            7 torch.Size([1, 64, 18, 44])
            8 torch.Size([1, 256, 18, 44])
            9 torch.Size([1, 256, 18, 44])
            10 torch.Size([1, 256, 18, 44])
            11 torch.Size([1, 256, 18, 44])
            12 torch.Size([1, 256, 18, 44])
            13 torch.Size([1, 256, 18, 44])
            14 torch.Size([1, 64, 16, 21])
            15 torch.Size([1, 64, 16, 21])
            16 torch.Size([1, 256, 16, 18])
            17 torch.Size([1, 256, 16, 18])
            18 torch.Size([1, 256, 16, 18])
            19 torch.Size([1, 256, 16, 18])
            20 torch.Size([1, 68, 4, 18])
            21 torch.Size([1, 68, 4, 18])
            22 torch.Size([1, 68, 4, 18])
            '''
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]     2 6 13 22的索引都是ReLU，但不包括全部的ReLU
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            # print(f.shape)
            '''
            torch.Size([1, 64, 4, 18])
            torch.Size([1, 128, 4, 18])
            torch.Size([1, 256, 4, 18])
            torch.Size([1, 68, 4, 18])
            '''
            global_context.append(f)

        x = torch.cat(global_context, 1)
        # print(x.shape) # [1, 516, 4, 18]
        x = self.container(x) # [1, 68, 4, 18]
        logits = torch.mean(x, dim=2)   # 这一步改了维度，[1, 68, 18]
        # logits = F.softmax(logits, dim=1)
        # print(logits.shape)

        return logits

def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):   # dropout_rate随机失活的比例，0.5可以大幅度改变网络结构

    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    """
    model.train()
    启用 BatchNormalization 和 Dropout
    
    model.eval()
    不启用 BatchNormalization 和 Dropout
    """
    if phase == True:   # 原来是 if phase == "train":  感觉写错了？
        return Net.train()
    else:
        return Net.eval()
