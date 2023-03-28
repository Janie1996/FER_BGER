
import os

import cv2
import cv2 as cv
import numpy as np
import xlrd
from PIL import Image
from scipy import io

BODY_PARTS= {"SpineBase":0,"SpineMid":1,"Neck":2,"Head":3,"ShoulderLeft":4,"ElbowLeft":5,"WristLeft":6,"HandLeft":7,"ShoulderRight": 8, "ElbowRight" :9, "WristRight" : 10, "HandRight" : 11, "HipLeft" : 12, "KneeLeft" : 13, "AnkleLeft" : 14, "FootLeft" : 15, "HipRight" : 16, "KneeRight" : 17, "AnkleRight" : 18, "FootRight" : 19, "SpineShoulder" : 20, "HandTipLeft" : 21, "ThumbLeft" : 22, "HandTipRight" : 23, "ThumbRight" : 24}

POSE_PAIRS =[ [3,2],[2,20],[20,4],[20,8],[20,1],[1,0],[4,5],[5,6],[6,7],[6,22],[7,21],[8,9],[9,10],[10,11],[10,24],[11,23],[0,12],[0,16],[12,13],[13,14],[14,15],[16,17],[17,18],[18,19] ]

size=64


def pose(x,length):

    i=0
    x_point=np.arange(28)
    y_point=np.arange(28)

    j = 0
    while (i < len(x) - 1):
        # start_x = int(x[i] + 1000)  # 32
        # start_y = int(x[i + 1] + 50)

        start_x = 1000 - (int(x[i]) + 80) * 2
        start_y = 1000 - (int(x[i + 1]) + 10) * 2
        x_point[j] = start_x
        y_point[j] = start_y
        j = j + 1

        i += 3

    min_x = np.min(x_point) - 5
    min_y = np.min(y_point) - 5
    max_x = np.max(x_point) + 5
    max_y = np.max(y_point) + 5
    scale_x = size / (max_x - min_x)
    scale_y = size / (max_y - min_y)
    scale = min(scale_x, scale_y)
    normalized_x = (x_point - min_x) * scale
    normalized_y = (y_point - min_y) * scale
    normalized_x = normalized_x.astype('int')
    normalized_y = normalized_y.astype('int')

    heatmaps=np.zeros([28,size,size])

    for i in range(28):
        # x = x_point[i]
        # y = y_point[i]
        x=normalized_x[i]   # x,y 分别保存了关节的位置坐标
        y=normalized_y[i]
        # if(i==7):
        #     print(x)
        heatmaps[i][x][y]=1     # 关节点位置赋值1，其余为0

    return heatmaps

def upper_pose(x,length):

    i=0
    x_point=np.arange(14)
    y_point=np.arange(14)

    j = 0
    while (i < 40):
        # start_x = int(x[i] + 1000)  # 32
        # start_y = int(x[i + 1] + 50)

        start_x = 1000 - (int(x[i]) + 80) * 2
        start_y = 1000 - (int(x[i + 1]) + 10) * 2
        x_point[j] = start_x
        y_point[j] = start_y
        j = j + 1

        i += 3

    min_x = np.min(x_point) - 5
    min_y = np.min(y_point) - 5
    max_x = np.max(x_point) + 5
    max_y = np.max(y_point) + 5
    scale_x = size / (max_x - min_x)
    scale_y = size / (max_y - min_y)
    scale = min(scale_x, scale_y)
    normalized_x = (x_point - min_x) * scale
    normalized_y = (y_point - min_y) * scale
    normalized_x = normalized_x.astype('int')
    normalized_y = normalized_y.astype('int')

    heatmaps=np.zeros([14,size,size])

    for i in range(14):
        # x = x_point[i]
        # y = y_point[i]
        x=normalized_x[i]   # x,y 分别保存了关节的位置坐标
        y=normalized_y[i]
        # if(i==7):
        #     print(x)
        heatmaps[i][x][y]=1     # 关节点位置赋值1，其余为0

    return heatmaps

# 获取每个视频每帧数据
def x_y(capture,length):


    heatmaps = np.zeros([length,28,size,size])
    readbook = xlrd.open_workbook(capture)
    sheet = readbook.sheet_by_index(2)  # 索引的方式，从0开始

    nrows = sheet.nrows  # 行

    if nrows<length:
        pre = []
        i = 0
        for j in range(1, nrows):
            x = sheet.row_values(j)
            heatmaps[i] = pose(x,length)
            i += 1
        pre = heatmaps[i - 1]
        for j in range(i, length):
            heatmaps[i] = pre
    else:
        j = 0
        y = np.linspace(0, nrows, length, False, True, int)[0]
        i = 0
        for num in range(1, nrows):
            x = sheet.row_values(num)
            if i in y:
                heatmaps[j] = pose(x,length)
                j += 1
            i += 1

    return heatmaps

def upper_x_y(capture,length):


    heatmaps = np.zeros([length,14,size,size])
    readbook = xlrd.open_workbook(capture)
    sheet = readbook.sheet_by_index(2)  # 索引的方式，从0开始

    nrows = sheet.nrows  # 行

    if nrows<length:
        pre = []
        i = 0
        for j in range(1, nrows):
            x = sheet.row_values(j)
            heatmaps[i] = upper_pose(x,length)
            i += 1
        pre = heatmaps[i - 1]
        for j in range(i, length):
            heatmaps[i] = pre
    else:
        j = 0
        y = np.linspace(0, nrows, length, False, True, int)[0]
        i = 0
        for num in range(1, nrows):
            x = sheet.row_values(num)
            if i in y:
                heatmaps[j] = upper_pose(x,length)
                j += 1
            i += 1

    return heatmaps


from Gesture.pre.new_Potion import get_one_channel


if __name__=="__main__":
    import cv2 as cv

    path = 'D:\\All_Xls_Files\\SW'
    allpath = []
    index = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            allpath.append(os.path.join(root, f))
            index += 1

    all_x_y = np.zeros([len(allpath), 2, length, size])
    all_potion = np.zeros([400, 28, size, size])

    for i in range(0,400):
        heatmaps = x_y(allpath[i])
        potion=get_one_channel(heatmaps)

        # for j in range(25):
        #     cv.imshow('OriginalPicture', potion[j])
        #     cv.waitKey()

        # cv.imshow('x',heatmaps[0])
        # cv.imshow('y',heatmaps[1])

        # cv.waitKey()
        all_potion[i] = potion
    io.savemat('D:\\All_Xls_Files\\SW_PoTion\\1_linear_potion_nor64.mat', {'test_data': all_potion})

    all_potion = np.zeros([400, 28, size, size])

    for i in range(400, 800):
        heatmaps = x_y(allpath[i])
        potion = get_one_channel(heatmaps)

        # for j in range(25):
        #     cv.imshow('OriginalPicture', potion[j])
        #     cv.waitKey()

        # cv.imshow('x',heatmaps[0])
        # cv.imshow('y',heatmaps[1])

        # cv.waitKey()
        all_potion[i-400] = potion
    io.savemat('D:\\All_Xls_Files\\SW_PoTion\\2_linear_potion_nor64.mat', {'test_data': all_potion})

    all_potion = np.zeros([len(allpath)-800, 28, size, size])

    for i in range(800, len(allpath)):
        heatmaps = x_y(allpath[i])
        potion = get_one_channel(heatmaps)

        # for j in range(25):
        #     cv.imshow('OriginalPicture', potion[j])
        #     cv.waitKey()

        # cv.imshow('x',heatmaps[0])
        # cv.imshow('y',heatmaps[1])

        # cv.waitKey()
        all_potion[i-800] = potion
    io.savemat('D:\\All_Xls_Files\\SW_PoTion\\3_linear_potion_nor64.mat', {'test_data': all_potion})