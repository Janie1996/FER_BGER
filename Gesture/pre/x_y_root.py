"""
    轨迹数据保存;
    N个关节数据形成单独通道
"""
import json
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
length=200

def pose1(x):

    i=0
    x_point=np.arange(28)
    y_point=np.arange(28)

    j=0
    while(i<len(x)-1):
        start_x = int(x[i] + 1000)  # 32
        start_y = int(x[i + 1] + 50)
        x_point[j] = start_x
        y_point[j] = start_y
        j=j+1


        i+=3

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

    heatmaps=np.zeros([28,2,size])

    for i in range(28):
        # x = x_point[i]
        # y = y_point[i]
        x=normalized_x[i]
        y=normalized_y[i]
        # if(i==7):
        #     print(x)
        heatmaps[i][0][x]+=i+1
        heatmaps[i][1][y]+=i+1
    #print(x)
    return heatmaps

def x_y1(capture):


    heatmaps = np.zeros([28, 2, length, size])

    readbook = xlrd.open_workbook(capture)
    sheet = readbook.sheet_by_index(2)  # 索引的方式，从0开始

    nrows = sheet.nrows  # 行
    if nrows < length:
        pre = []
        i = 0
        for j in range(1, nrows):
            x = sheet.row_values(j)
            heatmaps[:, :, i,:] = pose1(x)
            i += 1
        pre = heatmaps[:, :, i - 1,:]
        for j in range(i, length):
            heatmaps[:, :, i,:] = pre


    else:
        j=0
        y = np.linspace(0, nrows, length, False, True, int)[0]
        i = 0
        for num in range(1, nrows):
            x = sheet.row_values(num)
            if i in y:
                heatmaps[:, :, j,:] = pose1(x)
                j += 1
            i += 1

    return heatmaps


def save_label(action):

    path = 'D:\\All_Xls_Files\\'+action
    allpath = []

    for root, dirs, files in os.walk(path):
        for f in files:
            allpath.append(os.path.join(root, f))


    label = np.zeros([len(allpath)])

    dataset = {'Anger': 0, 'Anxiety': 1, 'Joy': 2, 'Neutral': 3, 'Panic Fear': 4, 'Pride': 5, 'Sadness': 6, 'Shame': 7}

    for i in range(len(allpath)):
        for key in dataset.keys():
            x = allpath[i].find(key)
            if (x != -1):
                label[i] = dataset[key]
                print(key)
                break

    io.savemat('D:\\All_Xls_Files\\label\\'+action+'.mat', {'test_label': label})


def main():

    save_label('SW')

    path = 'D:\\All_Xls_Files\\SW'
    allpath = []
    index = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            allpath.append(os.path.join(root, f))
            index += 1

    # all_x_y = np.zeros([200, 28, 2, length, size])
    all_x_y = np.zeros([len(allpath), 28, 2, length, size])

    for i in range(0, len(allpath)):
        heatmaps = x_y1(allpath[i])
        # cv.imshow('x',heatmaps[0])
        # cv.imshow('y',heatmaps[1])
        # cv.waitKey()
        all_x_y[i] = heatmaps
    io.savemat('D:\\All_Xls_Files\\SW_PoTion\\x_y_28_nor_200len.mat', {'test_data': all_x_y})



if __name__=="__main__":
    main()
