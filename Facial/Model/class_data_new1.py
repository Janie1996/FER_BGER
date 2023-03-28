#coding=utf-8
"""
    在dataset类里只保存视频路径，和对应的ID，label
    dataload在训练过程中读取每个视频内容
    优： 不占用大量内存
    缺： 每一轮训练都要从硬盘读取数据，耗时
"""
import sys
import os
##  之后所有的路径都是从项目位置直接写
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import os, sys, shutil
import random as rd
import cv2
from PIL import Image
import numpy as np
import random
import torch

import torch.utils.data as data

try:
    import cPickle as pickle
except:
    import pickle


cate2label = {'CK+':0,
              'ESVG': 1,
              'Enterface': 2
              }


cate2label = cate2label['CK+']



def get_single_video(path,height,width,data_transforms):


    vc = cv2.VideoCapture(path)  # 读入视频文件
    length=int(vc.get(7))
    video = torch.ones(length,3, height, width)

    for i in range(length):
        rval, frame = vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
        frame = data_transforms(frame).unsqueeze(0)
        video[i, :, :, :] = frame

    vc.release()
    return video



def load_video(video_root, video_list,C,K, rectify_label,transform):

    clips=list()

    with open(video_list, 'r') as imf:
        index = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video

            # ck+
            if cate2label==0:
                vc = cv2.VideoCapture(video_path + '/' + os.listdir(video_path)[0])
            else:
            # BAUM,enterface
                vc = cv2.VideoCapture(video_path)
            length= vc.get(7)
            if length > C*K:
                for i in range(1):     #数据增强，存3次
                    clips.append((video_path,label))
                index.append(np.ones(1) * id)

            else:
                clips.append((video_path,label))
                index.append(np.ones(1) * id)
        index = np.concatenate(index, axis=0)

    return  clips,index

def load_Allvideo(video_root, video_list,C,K, rectify_label,transform):

    clips=list()

    with open(video_list, 'r') as imf:
        index = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video

            # ck+
            if cate2label==0:
                vc = cv2.VideoCapture(video_path + '/' + os.listdir(video_path)[0])
            elif cate2label==2:
                # afew
                vc = cv2.VideoCapture(video_path+'.mp4')  # 读入视频文件
            else:
                # BAUM,enterface
                vc = cv2.VideoCapture(video_path)
            length= vc.get(7)
            if length > C*K:
                for i in range(1):     #数据增强，存3次
                    path=(video_path,label)
                    clips.append(get_video(path,C,K,transform))
                index.append(np.ones(1) * id)

            else:
                # clips.append((video_path,label))
                path=(video_path,label)
                clips.append(get_video(path,C,K,transform))
                index.append(np.ones(1) * id)
        index = np.concatenate(index, axis=0)

    return  clips,index



def get_video(video,C,K,transform):

    video_path,label=video
    clip=list()
    #ck+
    if cate2label==0:
        video = get_single_video(video_path+'/'+os.listdir(video_path)[0], 224, 224, transform)
    elif cate2label==2:
        video = get_single_video(video_path + '.mp4', 224, 224, transform)
    else:
        video = get_single_video(video_path, 224, 224, transform)
    # #print(video_path)

    if video.shape[0] > C * K:
        inter = int(video.shape[0] / C)
        for j in range(C):  # 将视频分为C个clip
            if j == C - 1:
                clips_index = range(j * inter, video.shape[0], 1)
            else:
                clips_index = range(j * inter, (j + 1) * inter, 1)
            str = random.sample(clips_index, K)
            str.sort()
            video_clip = torch.ones(K, 3, 224, 224)  # 每个clip共K个frame
            num = 0
            for k in str:
                video_clip[num] = video[k]
                num += 1
            clip.append((video_clip, label))
            #print(str)
    else:
        video_clip = torch.ones(C * K, 3, 224, 224)
        video_clip[:video.shape[0], :, :, :] = video
        for i in range(video.shape[0], C * K, 1):
            video_clip[i] = video[-1]
        for i in range(C):
            clip.append((video_clip[i * K:(i + 1) * K], label))
    return clip


def get_video1(video,C,K,transform):

    video_path,label=video
    clip=list()
    video_frame=list()
    if cate2label==0:
        video = get_single_video(video_path+'/'+os.listdir(video_path)[0], 224, 224, transform)
    elif cate2label==2:
        video = get_single_video(video_path + '.mp4', 224, 224, transform)
    else:
        video = get_single_video(video_path, 224, 224, transform)

    if video.shape[0] > C * K:
        inter = int(video.shape[0] / C)
        for j in range(C):  # 将视频分为C个clip
            if j == C - 1:
                clips_index = range(j * inter, video.shape[0], 1)
            else:
                clips_index = range(j * inter, (j + 1) * inter, 1)
            str = random.sample(clips_index, K)
            str.sort()
            video_clip = torch.ones(K, 3, 224, 224)  # 每个clip共K个frame
            num = 0
            for k in str:
                video_clip[num] = video[k]
                num += 1
            clip.append((video_clip, label))
            video_frame.append(str)
    else:
        video_clip = torch.ones(C * K, 3, 224, 224)
        video_clip[:video.shape[0], :, :, :] = video
        for i in range(video.shape[0], C * K, 1):
            video_clip[i] = video[-1]
        for i in range(C):
            clip.append((video_clip[i * K:(i + 1) * K], label))
    #print(video_path,video_frame)
    return clip


def sample_video(video,C,K,transform):

    video_path,label=video
    clip=list()
    #ck+

    if cate2label==0:
        video = get_single_video(video_path+'/'+os.listdir(video_path)[0], 224, 224, transform)
    elif cate2label==2:
        video = get_single_video(video_path + '.mp4', 224, 224, transform)
    else:
        video = get_single_video(video_path, 224, 224, transform)

    if video.shape[0] > C * K:
        inter = int(video.shape[0] / C)
        for j in range(C):  # 将视频分为C个clip
            if j == C - 1:
                clips_index = range(j * inter, video.shape[0], 1)
            else:
                clips_index = range(j * inter, (j + 1) * inter, 1)

            video_clip = torch.ones(K, 3, 224, 224)  # 每个clip共K个frame
            num = 0

            str = np.array([])
            seg = len(clips_index) / K
            for n in range(K):
                if int(seg * (n + 1)) >= len(clips_index):
                    str = np.append(str, clips_index[-1])
                    k=clips_index[-1]
                else:
                    str = np.append(str, clips_index[int(seg * n)])
                    k=clips_index[int(seg * n)]
                video_clip[num] = video[k]
                num += 1

            # str = random.sample(clips_index, K)
            # str.sort()
            # video_clip = torch.ones(K, 3, 224, 224)  # 每个clip共K个frame
            # num = 0
            #
            # for k in str:
            #     video_clip[num] = video[k]
            #     num += 1
            clip.append((video_clip, label))
    else:
        video_clip = torch.ones(C * K, 3, 224, 224)
        video_clip[:video.shape[0], :, :, :] = video
        for i in range(video.shape[0], C * K, 1):
            video_clip[i] = video[-1]
        for i in range(C):
            clip.append((video_clip[i * K:(i + 1) * K], label))
    return clip



class VideoDataset(data.Dataset):
    def __init__(self, video_root, video_list, C,K,rectify_label=None, transform=None):

        self.clips, self.index = load_video(video_root, video_list,C,K,rectify_label,transform)
        self.transform = transform
        self.C=C
        self.K=K

    def __getitem__(self, index):
        video=get_video(self.clips[index],self.C,self.K,self.transform)
        #video = sample_video(self.clips[index], self.C, self.K, self.transform)
        return video, self.index[index]   #  index 标明视频

    def __len__(self):
        return len(self.clips)


class VideoDataset1(data.Dataset):
    def __init__(self, video_root, video_list, C,K,rectify_label=None, transform=None):

        self.clips, self.index = load_video(video_root, video_list,C,K,rectify_label,transform)
        self.transform = transform
        self.C=C
        self.K=K

    def __getitem__(self, index):
        video=get_video1(self.clips[index],self.C,self.K,self.transform)
        return video, self.index[index]   #  index 标明视频

    def __len__(self):
        return len(self.clips)

class VideoDataset2(data.Dataset):
    def __init__(self, video_root, video_list, C,K,rectify_label=None, transform=None):

        self.clips, self.index = load_video(video_root, video_list,C,K,rectify_label,transform)
        self.transform = transform
        self.C=C
        self.K=K

    def __getitem__(self, index):
        video=sample_video(self.clips[index],self.C,self.K,self.transform)
        return video, self.index[index]   #  index 标明视频

    def __len__(self):
        return len(self.clips)

## 加载到内存中
class VideoDataset3(data.Dataset):
    def __init__(self, video_root, video_list, C,K,rectify_label=None, transform=None):

        self.clips, self.index = load_Allvideo(video_root, video_list,C,K,rectify_label,transform)
        self.transform = transform
        self.C=C
        self.K=K

    def __getitem__(self, index):
        #video=get_video(self.clips[index],self.C,self.K,self.transform)
        #video = sample_video(self.clips[index], self.C, self.K, self.transform)
        return self.clips[index], self.index[index]   #  index 标明视频

    def __len__(self):
        return len(self.clips)
