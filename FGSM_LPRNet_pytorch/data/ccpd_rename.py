import os
import cv2 as cv
import numpy as np

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"] 
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

img_path = './ccpd_base'
for root, dirs, files in os.walk(img_path):
       for name in files:
              plate = name.split('-')[4]
              plate = plate.split('_')
              label = ''
              label += provinces[int(plate[0])]
              for i in range(1, len(plate)):
                     label += ads[int(plate[i])]
              # print(label)
              label += '.jpg'

              frame = cv.imread(os.path.join(root, name))
              outputFile = os.path.join('./my_ccpd', label)
              # print(outputFile)
              cv.imwrite(outputFile, frame.astype(np.uint8))
              print(label, 'Done.')