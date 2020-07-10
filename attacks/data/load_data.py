from torch.utils.data import *
# from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}   # 建立字符和索引的映射

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None): # PreproFun为图像预处理，不提供的话用代码里的
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            # self.img_paths += [el for el in paths.list_images(img_dir[i])]  # 列表生成式，最终img_paths里面每个元素都是一张图片
            for root, dirs, files in os.walk(img_dir[i]):
                for name in files:
                    self.img_paths.append(os.path.join(root, name))

        random.shuffle(self.img_paths) # 改变图片顺序
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)    # 用opencv打开图像，返回的是numpy数组
        height, width, _ = Image.shape  # 高、宽、通道数
        if height != self.img_size[1] or width != self.img_size[0]: # 不符合要求的图片改成要求的尺寸
            Image = cv2.resize(Image, tuple(self.img_size))
        Image = self.PreprocFun(Image)  # 图像预处理

        basename = os.path.basename(filename)   # 去掉文件名前面的路径，只保留文件名
        imgname, suffix = os.path.splitext(basename)    # 分离扩展名
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c]) # 将字符的label对应到数字的label，存在列表里

        if len(label) == 8: # 8位车牌都是邪教
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"  # assert 0，直接返回错误信息

        return Image, label, len(label)

    def transform(self, img):   # 图像预处理，这里是图像归一化，范围为(-1,1)
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125  # 1 / 128
        img = np.transpose(img, (2, 0, 1)) # 这一步一直不知道是干嘛

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True
