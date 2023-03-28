"""
    将CK+保存的图片序列转化为mp4视频格式
"""

import os
import numpy as np
import cv2 as cv


def load_imgs_total_frame(video_root, video_list):

    with open(video_list, 'r') as imf:

        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of
            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending

            img = cv.imread(os.path.join(video_path, img_lists[0]) )
            x, y = img.shape[0:2]

            path='E:/video/'+video_label[0]
            file=os.path.exists(path)
            if (file==False):
                os.makedirs(path)

            outfile_name = path+'/'+img_lists[0][:-4] + '.mp4'

            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(outfile_name, fourcc, 15, (y,x))

            for frame in img_lists:
                img = cv.imread(os.path.join(video_path, frame))
                out.write(img)
            out.release()


if __name__=='__main__':
    arg_rootTrain = 'E:\\CK+\\extended-cohn-kanade-images\\cohn-kanade-images'
    arg_listTrain = 'list_train.txt'
    load_imgs_total_frame(arg_rootTrain, arg_listTrain)