'''
将一个文件夹的图片按 训练集：测试集=7:3 的比例分成两个文件夹
'''

import os
import random
import shutil
from  shutil import copy2

img_dirs = './plate'
trainfiles = os.listdir(img_dirs)
num_train = len(trainfiles)
index_list = list(range(num_train))
random.shuffle(index_list)  # 通过打乱下标来完成图像的随机分配
num = 0
trainDir = './pre_train'
validDir = './pre_test'

# 若不存在文件夹则创建
if not os.path.exists(trainDir):
    os.makedirs(trainDir)
if not os.path.exists(validDir):
    os.makedirs(validDir)
      
for i in index_list:
    fileName = os.path.join(img_dirs, trainfiles[i])
    if num < num_train*0.8:
        copy2(fileName, trainDir)
    else:
        copy2(fileName, validDir)
    num += 1
    if (num % 2000 == 0):
        print(num)