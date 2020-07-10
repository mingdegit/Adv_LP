# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

# darknet是YOLO作者自己写的一个深度学习框架

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n') # classes=['LP']

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "darknet-yolov3.cfg"   # 网络参数的配置文件
modelWeights = "lapi.weights"   # 模型权重文件

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

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3) # 这句框出了车牌，最后两个数字分别是颜色和粗细

    label = '%.2f' % conf # 预测自信度

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes)) # 类别号一定比类别数少，否则报错
        label = '%s:%s' % (classes[classId], label) # 要打印到结果图上的文字

    # Display the label at the top of the bounding box
    # 下面主要是画出标签的信息，车牌已经框出来了
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) # 获取文本框的大小
    # print(labelSize, baseLine)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
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
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# Process inputs
# winName = 'Deep learning object detection in OpenCV'
# cv.namedWindow(winName, cv.WINDOW_NORMAL)

# outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image) # VideoCapture类，打开图片、视频文件或视频流
    # outputFile = args.image[:-4]+'_yolo_out_py.jpg' # -4跳过.jpg后缀，目测jpeg得出问题
    outputFile = os.path.splitext(args.image)[0] + '_yolo_out_py.jpg' # os.path.splitext可分离文件名和扩展名，注意文件名是包括路径的
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    # outputFile = args.video[:-4]+'_yolo_out_py.avi'
    outputFile = os.path.splitext(args.image)[0] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0: # waitKey()返回值为被按键的值，如果超过指定时间则返回-1

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame: # 已经提取不出帧了，对于图片只能提取一次
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        # cv.waitKey(3000)
        break

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
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))
