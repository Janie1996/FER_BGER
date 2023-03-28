"""
    保存身体姿态轨迹数据，所有关节保留到一张图上
    pose()是用于ESVG；pose1()是用于BRED
"""
import os
import json
import cv2
import cv2 as cv
import numpy as np
import xlrd
from PIL import Image
from scipy import io

BODY_PARTS= {"SpineBase":0,"SpineMid":1,"Neck":2,"Head":3,"ShoulderLeft":4,"ElbowLeft":5,"WristLeft":6,"HandLeft":7,"ShoulderRight": 8, "ElbowRight" :9, "WristRight" : 10, "HandRight" : 11, "HipLeft" : 12, "KneeLeft" : 13, "AnkleLeft" : 14, "FootLeft" : 15, "HipRight" : 16, "KneeRight" : 17, "AnkleRight" : 18, "FootRight" : 19, "SpineShoulder" : 20, "HandTipLeft" : 21, "ThumbLeft" : 22, "HandTipRight" : 23, "ThumbRight" : 24}

POSE_PAIRS =[ [3,2],[2,20],[20,4],[20,8],[20,1],[1,0],[4,5],[5,6],[6,7],[6,22],[7,21],[8,9],[9,10],[10,11],[10,24],[11,23],[0,12],[0,16],[12,13],[13,14],[14,15],[16,17],[17,18],[18,19] ]

size=200
length=100

def pose(x):

    i=2
    x_point=np.arange(25)
    y_point=np.arange(25)

    j=0
    while(i<len(x)-1):
        start_x=int((1-float(x[i]))*372)  #32
        start_y=int((1-float(x[i+1]))*372)
        x_point[j]=start_x
        y_point[j]=start_y
        j=j+1

        i+=6

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

    heatmaps=np.zeros([2,size])

    for i in range(25):
        # x = x_point[i]
        # y = y_point[i]
        x=normalized_x[i]
        y=normalized_y[i]
        # if(i==7):
        #     print(x)
        heatmaps[0][x]+=i+1
        heatmaps[1][y]+=i+1
    #print(x)
    return heatmaps


def pose1(x):

    i=0
    x_point=np.arange(28)
    y_point=np.arange(28)

    j=0
    while(i<len(x)-1):
        start_x=int(x[i]+1000)  #32
        start_y=int(x[i+1]+50)
        x_point[j]=start_x
        y_point[j]=start_y
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

    heatmaps=np.zeros([2,size])

    for i in range(28):
        # x = x_point[i]
        # y = y_point[i]
        x=normalized_x[i]
        y=normalized_y[i]
        # if(i==7):
        #     print(x)
        heatmaps[0][x]+=i+1
        heatmaps[1][y]+=i+1
    #print(x)
    return heatmaps

def get_info(path):
    video_dataset = []
    dir_name = []  # 保存文件夹名字
    all_video = []  # 文件夹下文件名字
    file_names=[]
    index = -1
    for root, dirs, files in os.walk(path):
        if (root == path):
            dir_name = dirs
        if (files):
            all_video.append(files)
            index += 1
        for f in files:
            video_dataset.append(os.path.join(root, f))
            file_names.append(f[:-3]+'mp4')
            # key_video.append(os.path.join(os.path.join(endpath, dir_name[index]), f))
    return video_dataset,file_names

def x_y(capture):

    file = open(capture, 'r')
    count = len(file.readlines())
    # heatmaps = np.zeros([2,length,size])
    heatmaps = np.zeros([2, size, length])
    file.seek(0)

    if count<length:
        pre = []
        for i in range(length):
            x = file.readline()
            x = x.split(';')
            if (len(x) == 1):
                heatmaps[:, :, i] = pre
            else:
                heatmaps[:, :, i] = pose(x)
            pre = heatmaps[:, :, i]
    else:
        j=0
        y = np.linspace(0, count, length, False, True, int)[0]
        for i in range(count):
            x = file.readline()
            x = x.split(';')
            if i in y:
                heatmaps[:, :, j] = pose(x)
                j+=1

    return heatmaps

def x_y1(capture):

    # heatmaps = np.zeros([2,length,size])
    heatmaps = np.zeros([2, size, length])

    readbook = xlrd.open_workbook(capture)
    sheet = readbook.sheet_by_index(2)  # 索引的方式，从0开始

    nrows = sheet.nrows  # 行


    if nrows<length:
        pre = []
        i=0
        for j in range(1,nrows):
            x = sheet.row_values(j)
            heatmaps[:, :, i] = pose1(x)
            i+=1
        pre = heatmaps[:, :, i-1]
        for j in range(i,length):
            heatmaps[:, :, i]=pre
    else:
        j=0
        y = np.linspace(0, nrows, length, False, True, int)[0]
        i = 0
        for num in range(1,nrows):
            x = sheet.row_values(num)
            if i in y:
                heatmaps[:, :, j] = pose1(x)
                j+=1
            i+=1

    return heatmaps


def main():
    path='D:\\All_Xls_Files\\SW'
    video_dataset = []
    index = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            video_dataset.append(os.path.join(root, f))
            index += 1


    all_x_y = np.zeros([len(video_dataset), 2, size, length])

    for i in range(0, len(video_dataset)):
        heatmaps = x_y1(video_dataset[i])
        # cv.imshow('x',heatmaps[0])
        # cv.imshow('y',heatmaps[1])
        # cv.waitKey()
        print(i)
        all_x_y[i] = heatmaps
    io.savemat('D:\\All_Xls_Files\\SW_PoTion\\x_y_nor.mat', {'test_data': all_x_y})


if __name__=="__main__":

    main()
