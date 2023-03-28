"""
    混淆矩阵图画法：CK+  ENTERFACE05   BRED

"""

from __future__ import division
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plotCM(classes, matrix, savname):

    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    matrix*=100
    matrix = np.round(matrix, 2)  # 小数点位数
    font1 = {
             'color': 'white',
             }

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=plt.cm.get_cmap('gray_r'))
    #fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(i==j):
                ax.text(i, j, str(matrix[j][i]), va='center', ha='center', fontdict = font1)
            else:
                ax.text(i, j, str(matrix[j][i]), va='center', ha='center')

    ax.set_xticklabels([''] + classes, rotation=45)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_yticklabels([''] + classes)

    plt.savefig(savname,dpi=400)

