# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

# darknet是YOLO作者自己写的一个深度学习框架

import cv2 as cv
import numpy as np
import os

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

# Load names of classes
classesFile = "./Licence_plate_detection/classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n') # classes=['LP']

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "./Licence_plate_detection/darknet-yolov3.cfg"   # 网络参数的配置文件
modelWeights = "./Licence_plate_detection/lapi.weights"   # 模型权重文件

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights) # Reads a network model stored in Darknet model files.
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # print(layersNames)

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    # net.getUnconnectedOutLayers(): Returns indexes of layers with unconnected outputs.
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()] # ['yolo_82', 'yolo_94', 'yolo_106']

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, border):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = [] # 存储边界
    for out in outs:
        # print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores) # 找到最大值的索引，本来就只有LP类。。所以肯定识别索引都是0
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            # if detection[4] > confThreshold:
            #     print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
            #     print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                # 通过中心找到边界
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold) # 非最大抑制，主要是从boxes中找到我们需要的box
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        border.extend([left, top, left + width, top + height])


def crop_img(image_path):
    '''
    裁剪出车牌图像
    '''
    for root, dirs, files in os.walk(image_path):
        for filename in files:
            # Open the image file
            frame = cv.imread(os.path.join(root, filename))

            # Create a 4D blob from a frame.
            # blobFromImage对图片进行预处理
            # frame为图片，1/255为图片减去mean后对图片进行缩放的系数,size是我们神经网络在训练的时候要求输入的图片尺寸
            # mean：减去平均值（mean）：为了消除同一场景下不同光照的图片，对我们最终的分类或者神经网络的影响，
            #       我们常常对图片的R、G、B通道的像素求一个平均值，然后将每个像素值减去我们的平均值，这样就可以得到像素之间的相对值，就可以排除光照的影响。这里设置均值为0，我们只做归一化
            # swapRB：OpenCV中认为我们的图片通道顺序是BGR，但是我平均值假设的顺序是RGB，所以如果需要交换R和G，那么就要使swapRB=true
            blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net)) # getOutputsNames获得输出层的名字 ['yolo_82', 'yolo_94', 'yolo_106']
            # len(outs) = 3, type(outs[0]) = numpy数组

            # Remove the bounding boxes with low confidence
            border = []
            postprocess(frame, outs, border)
            try:
                left, bottom, right, top = border
            except:
                print(filename, 'fail.')
            else:
                frame = frame[bottom:top, left:right] # 裁剪坐标为[y0:y1, x0:x1]
                # Write the frame with the detection boxes
                outputFile = os.path.join('./data/my_ccpd_aft_crop', filename)
                cv.imwrite(outputFile, frame.astype(np.uint8))
                print(filename, 'Done')
