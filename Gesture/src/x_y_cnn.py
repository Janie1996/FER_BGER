import os
from scipy import io
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from Gesture.pre.x_y_root_size_length import x_y1

from Gesture.src.shuffle import shuffle


# two-layer traditional model
class CNN(nn.Module):
    def __init__(self,input_channel,emotion):
        super(CNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
        )
        # self.se1=SELayer(128)
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

#作用：如果要预测K个类别，在卷积特征抽取部分的最后一层卷积层，就会生成K个特征图，然后通过全局平均池化就可以得到 K个1×1的特征图，将这些1×1的特征图输入到softmax layer之后，每一个输出结果代表着这K个类别的概率（或置信度 confidence），起到取代全连接层的效果。
# 优点：      和全连接层相比，使用全局平均池化技术，对于建立特征图和类别之间的关系，是一种更朴素的卷积结构选择。
#     全局平均池化层不需要参数，避免在该层产生过拟合。
#     全局平均池化对空间信息进行求和，对输入的空间变化的鲁棒性更强
        self.pool=nn.AdaptiveAvgPool2d(1)#functional.adaptive_avg_pool2d(1)

        self.fc=nn.Linear(128,emotion)
        self.out=nn.Softmax()

    def forward(self,x):

        x=self.block1(x)
        # x=self.se1(x)
        x=self.block2(x)
        # x = self.se2(x)

        x=self.pool(x)
        x=x.view(x.size(0), -1)
        out=self.fc(x)
        out=self.out(out)
        return out

# ACCNN model
class proposed(nn.Module):
    def __init__(self,input_channel,emotion):
        super(proposed, self).__init__()

        self.channel1=nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.channel2=nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=3,stride=1,padding=1,groups=input_channel),
            # nn.BatchNorm2d(50),
            # nn.ReLU(),
            nn.Conv2d(in_channels=input_channel,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),


        )
        self.att=SELayer(128)
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool=nn.AdaptiveAvgPool2d(1)

        self.fc=nn.Linear(128,emotion)
        self.out=nn.Softmax()

    def forward(self,x):
        x1=self.channel1(x)
        x2=self.channel2(x)
        x2=self.att(x2)
        x2=self.block2(x2)

        #x=self.relu(self.bn(x2+x1))
        x=self.pool(x1+x2)
        x=x.view(x.size(0), -1)
        out=self.fc(x)
        #out=self.out(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

import time
import numpy as np

def main(action):
    emotion_num=8
    size = 64
    length = 300
    joint_num = 28

    x,train_index,test_index = splitTrainTest(action,emotion_num)

    # 获取数据轨迹数据
    path = 'D:\\Emilya\\All_Xls_Files\\'+action
    allpath = []
    index = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            allpath.append(os.path.join(root, f))
            index += 1

    y = np.zeros([len(allpath), joint_num, 2, size, length])
    # all_x_y = np.zeros([len(allpath), 28, 2, length, size])

    for i in range(0, len(allpath)):
        heatmaps = x_y1(allpath[i],joint_num,size,length)
        y[i] = heatmaps

    # y= read_mat.read_mat('D:\\All_Xls_Files\\SW_PoTion\\x_y_28_nor.mat')['test_data']


    m = np.zeros([len(x), joint_num * 2, size,length])
    for i in range(joint_num):
        m[:, i * 2:(i + 1) * 2, :, :] = y[:, i, :, :, :]

    testNum = len(test_index)
    trainNum = len(train_index)

    x_train = np.zeros([trainNum, joint_num * 2, size,length])
    y_train = np.zeros((trainNum,), dtype=np.int)
    for i in range(trainNum):
        x_train[i] = m[train_index[i]]
        y_train[i] = x[train_index[i]]

    x_test = np.zeros([testNum, joint_num * 2,  size,length])
    y_test = np.zeros((testNum,), dtype=np.int)
    for i in range(testNum):
        x_test[i] = m[test_index[i]]
        y_test[i] = x[test_index[i]]

    del x, y, m
    [x_train, y_train] = shuffle(x_train, y_train)

    # 定义网络
    net = proposed(joint_num * 2, emotion_num)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(net.parameters(), lr=1e-3)
    # optimizer_ft=optim.SGD(net.parameters(),lr=1e-1)
    # optimizer_ft=optim.RMSprop(net.parameters(),lr=1e-3)
    # optimizer_ft=optim.Adagrad(net.parameters(),lr=1e-1)
    # Decay LR by a factor of 0.1 every 7 epochs  修改学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    n_epochs = 200
    best_test = 0.0

    best_pre = []
    # file = open('loss.txt', 'w')
    # file1 = open('acc.txt', 'w')

    pro = nn.Softmax()

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs))
        print('-' * 10)
        train(net,optimizer_ft,exp_lr_scheduler,criterion,trainNum,x_train,y_train)
        epoch_acc, endpre=eval(net,testNum,x_test,y_test)

        if epoch_acc > best_test:
            best_pre = endpre
            best_test = epoch_acc

    # file1.close()
    # file.close()
    print(action)
    print("best", best_test)
    np.save("D:\\All_Xls_Files\\PoTion\\"+action+".npy", best_pre)


#加载数据集标签；划分数据集
def splitTrainTest(action,emotion_num):
    x = io.loadmat('D:\\Emilya\\All_Xls_Files\\label\\'+action+'.mat')['test_label'][0].astype("u1")

    # 统计每个情感类别样本数
    y_num = np.zeros(emotion_num)  # .astype("u1")
    for i in range(emotion_num):
        y_num[i] = int(np.sum(x == i))

    #  4折交叉验证；划分出测试集与训练集
    start = 0
    end = y_num[0]
    for i in range(emotion_num-1):
        y = np.linspace(start, end, int((end - start) / 10), False, True, int)[0]
        if (i == 0):
            test_index = y
        else:
            test_index = np.append(test_index, y, 0)
        print(y)
        start += y_num[i]
        end += y_num[i + 1]
    y = np.linspace(start, end, int((end - start) / 10), False, True, int)[0]
    test_index = np.append(test_index, y, 0)


    all_index = np.linspace(0, len(x) - 1, len(x), True, False, int)

    train_index = set(all_index) ^ set(test_index)
    train_index = list(train_index)
    train_index = np.array(train_index)

    return x,train_index,test_index


def train(net,optimizer_ft,exp_lr_scheduler,criterion,trainNum,x_train,y_train):

    exp_lr_scheduler.step()
    iter_per_epoch=10   # batch
    running_loss = 0.0
    running_corrects = 0.0
    tatal = 0.0

    # Iterate over data.
    net.train(True)
    indices = np.linspace(0, trainNum, iter_per_epoch)
    indices = indices.astype('int')

    for iter_i in range(iter_per_epoch - 1):
        inputs = x_train[indices[iter_i]:indices[iter_i + 1]]
        labels = y_train[indices[iter_i]:indices[iter_i + 1]]

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.float().cuda()
        labels = labels.long().cuda()

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer_ft.step()

        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.cpu().item() * labels.size(0)
        tatal = tatal + labels.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = float(running_corrects) / float(tatal)
    running_loss = float(running_loss) / float(tatal)
    print('{} : {:.6f}'.format('loss', running_loss))
    print('{} : {:.6f}'.format('acc', epoch_acc))

    # file.write(str(running_loss))
    # file.write('\n')
    # file1.write(str(epoch_acc))
    # file1.write('\n')

def eval(net,testNum,x_test,y_test):


    endpre = []

    net.train(False)
    running_corrects = 0.0
    tatal = 0.0


    indices = np.linspace(0, testNum, 10)
    indices = indices.astype('int')
    for iter_i in range(10 - 1):
        inputs = x_test[indices[iter_i]:indices[iter_i + 1]]
        labels = y_test[indices[iter_i]:indices[iter_i + 1]]

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.float().cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outputs = net(inputs)
            # outputs = net(inputs)
        _, preds = torch.max(outputs.data, 1)
        # statistics
        tatal = tatal + labels.size(0)
        running_corrects += torch.sum(preds == labels.data)
        # outputs = pro(outputs)
        if (iter_i == 0):
            endpre = outputs.data.cpu().numpy()
        else:
            endpre = np.vstack((endpre, outputs.data.cpu().numpy()))
            # endpre = np.append(endpre, preds.data.cpu().numpy())

    epoch_acc = float(running_corrects) / float(tatal)
    print('{} Acc: {:.4f}'.format('test', epoch_acc))

    return epoch_acc,endpre

if __name__=='__main__':
    actions = ['BS', 'KD', 'Lf', 'MB', 'SD', 'SW', 'Th', 'WH']
    for i in range(0,len(actions)):
        main(actions[i])
    for i in range(0, len(actions)):
        main(actions[i])