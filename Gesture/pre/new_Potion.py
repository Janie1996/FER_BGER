import os, sys
import numpy as np
import scipy.io as sio
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib.image as mpimg
import imageio
import h5py
import random
from PIL import Image
import json
from astropy.modeling.models import Gaussian2D
from numpy.random import randint
from matplotlib import animation
from multiprocessing import Pool


def get_one_channel(data_matrix):

    time_interval_T = data_matrix.shape[0]
    potion = []

    # 关节

    for i in range(0, data_matrix.shape[1]):
        # init
        curr_joint = data_matrix[:, i, :, :]
        [ori_time_t, x, y] = curr_joint.shape

        curr_Potion = np.zeros([ x, y])

        # 时间
        for T in range(0, time_interval_T):
            # 通道
            index = 0


            # calculate the clip of the original time series
            time_original = round(ori_time_t / time_interval_T * T)

            # update Potion representation, where index - 1 = [1 - (t-1)/(T-1)] and index = [(t-1)/(T-1)]

#*(T+1)
            curr_Potion += curr_joint[T, :, :] * (T / (time_interval_T-1))
            #curr_Potion += curr_joint[T, :, :] * ((T - 1) / (time_interval_T - 1))

        # the data from time T is added to the last channel
        # curr_Potion+= curr_joint[-1, :, :]  # Sj[x,y,c]

        # normalize each channel
        # for i_channel in range(np.shape(curr_Potion)[0]):
        #     curr_Potion[i_channel, :, :] = curr_Potion[i_channel, :, :] / (
        #                 sys.float_info.epsilon + np.max(curr_Potion[i_channel, :, :]))  # Uj[x,y,c]

        potion.append(curr_Potion)

    return np.array(potion)


'''
input:
    data_matrix: non-negative heat map of each joint, in shape [n_frames, n_joints, x, y]
    channel: required channel number, e.g., 2,3,4,5...
    time_interval_T: required sample T number, channel<= T <= frames

output:
    potion: Potion matrix in shape [n_joints, n_channel, x, y]
'''


def get_potion(data_matrix, channel):
    time_interval_T=data_matrix.shape[0]
    potion = []

    # 关节

    for i in range(0,data_matrix.shape[1]):
        # init
        curr_joint = data_matrix[:,i,:,:]
        [ori_time_t, x, y] = curr_joint.shape
        
        curr_Potion = np.zeros([channel, x, y])
        
        # slit the time_interval_T into (channel - 1) subsections
        time_arr = np.arange(1, time_interval_T, (time_interval_T - 1) / (channel - 1))
        # add the time_interval_T itself 
        time_arr = np.append(time_arr, time_interval_T)

        # 时间
        for T in range(1, time_interval_T):
            #通道
            index = 0
            
            # get the subsections (index) where the current time lies
            for k in np.nditer(time_arr):
                if T < k:
                    break
                index += 1

            # calculate the clip of the original time series
            time_original = round(ori_time_t / time_interval_T * T)

            # update Potion representation, where index - 1 = [1 - (t-1)/(T-1)] and index = [(t-1)/(T-1)]

            curr_Potion[index - 1, :, :] += curr_joint[time_original-1, :, :] \
                                                * (1 - (T - time_arr[index - 1]) / (time_arr[index] - time_arr[index - 1]))

            curr_Potion[index , :, :] += curr_joint[time_original-1, :, :] \
                                                * ((T - time_arr[index - 1]) / (time_arr[index] - time_arr[index - 1]))

        # the data from time T is added to the last channel
        curr_Potion[-1, :, :] += curr_joint[-1, :, :]   #Sj[x,y,c]

        # normalize each channel
        for i_channel in range(np.shape(curr_Potion)[0]):
            curr_Potion[i_channel, :, :] = curr_Potion[i_channel, :, :] / (sys.float_info.epsilon+np.max(curr_Potion[i_channel, :, :]))     #Uj[x,y,c]
            
        potion.append(curr_Potion)
        
    return np.array(potion)

'''
input:
    Potion: Potion matrix in shape [n_joints, channel, x, y]
output:
    r_Intensity: Intensity matrix in shape [n_joints, x, y]
'''
def get_intensity(Potion):
    intensity = []
    
    for i in range(0,Potion.shape[0]):
        curr_potion = Potion[i,:,:,:]
        curr_Intensity = np.zeros([curr_potion.shape[1], curr_potion.shape[2]])
        for i_channel in range(np.shape(curr_potion)[0]):
            curr_Intensity += curr_potion[i_channel, :, :]   #Ij[x,y]
        intensity.append(curr_Intensity)
    return np.array(intensity)



'''
input:
    Potion: Potion matrix in shape [n_joints, channel, x, y]
    Intensity: Intensity matrix in shape [n_joints, x, y]

output:
    r_Normalized: Normalized matrix in shape [n_joints, channel, x, y]
'''
def get_normalized_potion(Potion, Intensity):
    normalized = []
    
    for i in range(0,Potion.shape[0]):
        curr_potion = Potion[i,:,:,:]
        curr_intensity = Intensity[i,:,:]
        
        curr_normalized = np.zeros(curr_potion.shape)
    #         hyperparameter, original set to 1
        epsilon = 1
        for i_channel in range(np.shape(curr_normalized)[0]):   #Nj[x,y,c]
            curr_normalized[i_channel, :, :] = curr_potion[i_channel, :, :] / (epsilon + curr_intensity)
        
        normalized.append(curr_normalized)       
    return np.array(normalized)


'''
Stack three representations and resize to the given shape
Potion: [n_joints, channel, height, width]
Intensity: [n_joints, height, width]
Normalized_potion: [n_joints, channel, height, width]
'''
def get_descriptor(heatmaps,C):
    potion = get_potion(heatmaps, C)
    intensity = get_intensity(potion)
    normalized= get_normalized_potion(potion,intensity)
    potion=np.reshape(potion,(potion.shape[0]*potion.shape[1],potion.shape[2],potion.shape[3]))
    normalized=np.reshape(normalized,(normalized.shape[0]*normalized.shape[1],normalized.shape[2],normalized.shape[3]))
    descriptor=np.vstack((potion,intensity,normalized))
    return descriptor


'''
Define the mapping from semantic names to labels
'''
# dict_action = {
#     "baseball_pitch": 1,
#     "clean_and_jerk": 2,
#     "pull_up": 3,
#     "pullup": 3,
#     "strumming_guitar": 4,
#     "strum_guitar": 4,
#     "baseball_swing": 5,
#     "golf_swing": 6,
#     "push_up": 7,
#     "pushup": 7,
#     "tennis_forehand": 8,
#     "bench_press": 9,
#     "jumping_jacks": 10,
#     "sit_up": 11,
#     "situp": 11,
#     "tennis_serve": 12,
#     "bowl": 13,
#     "jump_rope": 14,
#     "squats": 15,
#     "squat": 15,
# }

if __name__ == '__main__':
    import cv2 as cv
    x=np.zeros([6,1,3,3])
    x[0,:,0,0]=1
    x[1,:,0,1]=1
    x[2,:,0,2]=1
    x[3,:,0,2]=1
    x[4,:,0,1]=1
    x[5,:,0,0]=1
    m=get_potion(x,2)
    n=get_one_channel(x)
    print(m[0][0])
    print(m[0][1])
    # print(m[0][2])

    x=np.zeros([6,1,3,3])
    x[0,:,0,2]=1
    x[1,:,0,1]=1
    x[2,:,0,0]=1
    x[3,:,0,0]=1
    x[4,:,0,1]=1
    x[5,:,0,2]=1
    n=get_potion(x,2)
    print(n[0][0])
    print(n[0][1])


    # m=np.swapaxes(m,1,2)
    # m=np.swapaxes(m,2,3)
    # cv.imshow('OpenPose using OpenCV', m[0])
    # cv.waitKey(0)

    n=get_descriptor(x,2)
    print(m)
