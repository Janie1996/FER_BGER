"""
    特征提取：  resnet18--2维卷积；输入视频流，转换为3个视觉角度，但共享模型参数
    情感识别：  一层LSTM； 输入clip级视频特征

"""


import argparse
import os

import torch.nn as nn
import pretrainedmodels
import torch
from Facial.src import util
from Facial.Model import attention_2D_2dim
from Facial.Model.load_dataset import LoadData
from Facial.Model.load_parameter import savecheckpoint
import torchvision.models as models

import numpy as np

class jointLoss(nn.Module):
    def __init__(self):
        super(jointLoss, self).__init__()
        #self.loss1=nn.MSELoss()
        self.loss2=nn.CrossEntropyLoss()
        self.loss3=nn.CrossEntropyLoss()
        self.loss4=nn.CrossEntropyLoss()
    def forward(self,p1,p2,p3,target ):
        #x1=self.loss1(p1,p2)
        x2=self.loss2(p1,target)
        x3=self.loss3(p3,target)
        x4=self.loss4(p2,target)
        return x2+x3+x4 # x1


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dir',default='/DATA/Storage/WeiJie/Dataset/CK+/video')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--C', default=5, type=int,
                    help='the clip number of a video')
parser.add_argument('--K', default=2, type=int,
                    help='the frame number of a clip')


args = parser.parse_args()

def main(times):

    global args,state_dict
    best_pre = 0
    ''' Load data '''
    dir_train =  'E:/VideoEmotionDatabase/CK+/video'
    list_train = 'Facial/Data/CK+/train'+str(times) +'.txt'
    batch_train=8

    dir_eval =  'E:/VideoEmotionDatabase/CK+/video'
    list_eval = 'Facial/Data/CK+/eval'+str(times) +'.txt'
    batch_eval=16

    train_loader,val_loader=LoadData(dir_train,list_train,batch_train,dir_eval,list_eval,batch_eval,args.C,args.K)


    # ''' Load model '''
    model = attention_2D_2dim.resnet18(batch=batch_train,C=args.C,K=args.K,num_classes=7)
    args1 = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet').state_dict()
    model_state_dict = model.state_dict()

    for key in args1:
        # if 'conv' in key or 'downsample.0' in key:
        #     args1[key] = args1[key].unsqueeze(2)
        if key in model_state_dict:
            model_state_dict[key] = args1[key]

    model.load_state_dict(model_state_dict)

    del args1,model_state_dict

    ''' Loss & Optimizer '''
    # criterion = nn.CrossEntropyLoss()
    criterion=jointLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    if args.evaluate == True:
        validate(val_loader, model)
        return

    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)

        prec1, prec2, prec3, prec4= validate(val_loader, model)


        prec1 = max(prec1,prec2,prec3,prec4)

        is_best = prec1 > best_pre
        if is_best:
            best_pre = max(prec1, best_pre)
            state_dict = model.state_dict()
        print("epoch:",epoch,"  best:",best_pre)
    print('best_pre:',best_pre)

    savecheckpoint({
        'epoch': args.epochs,
        'state_dict': state_dict,
        'prec1': best_pre,
    },'E:/PycharmProjects/facial expression/experiment/CK+/model/10')


def train(train_loader, model, criterion, optimizer, epoch):

    losses = util.AverageMeter()
    topframe = util.AverageMeter()
    topVideo = util.AverageMeter()

    pred_p1=util.AverageMeter()
    pred_p2=util.AverageMeter()
    pred_p3 = util.AverageMeter()

    output_store_fc = []
    target_store = []
    index_vector = []

    model.train()
    model.cuda()

    all=0
    right=0

    for i, (video,index) in enumerate(train_loader):

        target = video[0][1]    # 类别标签
        input = video[0][0].unsqueeze(0)   # frame 【1，batch，length，channel，】
        indexs = index     # 视频编号
        for j in range(1, args.C):
            #target = torch.cat([target, video[j][1]], 0)
            input = torch.cat([input, video[j][0].unsqueeze(0)], 0)
            #indexs= torch.cat([indexs,index],0)

        # input 【clip_length，batch，frame_length，channel
        target=target.cuda()
        input= input.view(-1, args.K, 3, 224, 224).permute(0, 2, 1, 3, 4).cuda()   #【clip_length*batch，channel，length，height，width】

        # clips = torch.ones([args.C, 8, 3,2,224,224])
        #
        # for i in range(args.C):
        #     clips[i] = input1[i * 8:(i + 1) * 8, :,:,:,:]
        # clips = clips.transpose(2,3).cuda()



        # result=input.permute(0, 2,  3, 4, 1 ).cpu().numpy()
        # from matplotlib import pyplot as plt
        # grid = plt.GridSpec(1, 14)
        #
        # for i in range(14):
        #     plt.subplot(grid[0, i]).imshow(result[i][0])
        # plt.show()

        p1,p2,p3,pred_score = model(input)

        loss = criterion(p1,p2,p3,target)
        # loss = criterion(pred_score,target)
        loss = loss.sum()

        _, preds = torch.max(pred_score.data, 1)
        corrects = float(torch.sum(preds == target.data))/float(target.size(0))
        right+= torch.sum(preds == target.data)
        all+= target.size(0)


        losses.update(loss.item(), target.size(0))
        topframe.update(corrects, target.size(0))

        _, preds = torch.max(p1.data, 1)
        corrects = float(torch.sum(preds == target.data)) / float(target.size(0))
        pred_p1.update(corrects, target.size(0))

        _, preds = torch.max(p2.data, 1)
        corrects = float(torch.sum(preds == target.data)) / float(target.size(0))
        pred_p2.update(corrects, target.size(0))

        #
        output_store_fc.append(pred_score.cpu())
        target_store.append(target.cpu())
        index_vector.append(indexs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {topframe.val:.3f} ({topframe.avg:.3f})\t'
        #         .format(epoch, i, len(train_loader), loss=losses, topframe=topframe))

    del input,video,loss

    #print("train:",float(right)/float(all))
    index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
    index_matrix = []
    for i in range(int(max(index_vector)) + 1):
        index_matrix.append(index_vector == i)  # 找出相同视频

    index_matrix = torch.stack(index_matrix, dim=0).float()  # [21570]  --->  [380, 21570]

    output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]

    target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]

    pred_matrix_fc = index_matrix.mm(output_store_fc)  # [380,21570] * [21570, 7] = [380,7]
    target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
        index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]

    _, preds = torch.max(pred_matrix_fc.data, 1)
    corrects = float(torch.sum(preds == target_vector.data)) / float(target_vector.size(0))
    topVideo.update(corrects, i + 1)
    print('Loss:', losses.avg)
    print('Train *Prec@Video {topVideo.avg:.3f}   *Prec@Frame {topframe.avg:.3f} '.format(topVideo=topVideo, topframe=topframe))
    # print('feature {pred_p1.avg:.3f}   emotion {pred_p2.avg:.3f}  '.format(pred_p1=pred_p1,pred_p2=pred_p2))

def validate(val_loader,model):

    all_target=[]
    all_pre=[]

    topframe = util.AverageMeter()
    topVideo = util.AverageMeter()
    pred_p1=util.AverageMeter()
    pred_p2=util.AverageMeter()
    pred_p3 = util.AverageMeter()

    # 融合
    end=util.AverageMeter()
    end1=util.AverageMeter()
    end2=util.AverageMeter()
    end3=util.AverageMeter()
    end4=util.AverageMeter()

    output_store_fc = []
    target_store = []
    index_vector = []

    model.eval()
    model.cuda()
    with torch.no_grad():
        for i, (video, index) in enumerate(val_loader):

            target = video[0][1]
            input = video[0][0].unsqueeze(0)
            indexs = index
            for j in range(1, args.C):
                #target = torch.cat([target, video[j][1]], 0)
                input = torch.cat([input, video[j][0].unsqueeze(0)], 0)
                #indexs = torch.cat([indexs, index], 0)
            target=target.cuda()
            input = input.view(-1, args.K, 3, 224, 224).permute(0, 2, 1, 3, 4).cuda()

            p1,p2,p3,pred_score = model(input)

            _, preds = torch.max(pred_score.data, 1)
            corrects = float(torch.sum(preds == target.data)) / float(target.size(0))

            topframe.update(corrects, target.size(0))

            _, preds = torch.max(p1.data, 1)
            corrects = float(torch.sum(preds == target.data)) / float(target.size(0))
            pred_p1.update(corrects, target.size(0))
            # p1=preds

            _, preds = torch.max(p2.data, 1)
            corrects = float(torch.sum(preds == target.data)) / float(target.size(0))
            pred_p2.update(corrects, target.size(0))
            # p2=preds

            _, preds = torch.max(p3.data, 1)
            corrects = float(torch.sum(preds == target.data)) / float(target.size(0))
            pred_p3.update(corrects, target.size(0))
            # p3 = preds

            if (i == 0):
                all_target=list(target.data.cpu().numpy())
                all_pre=list(preds.cpu().numpy())
            else:
                all_target=all_target+list(target.data.cpu().numpy())
                all_pre=all_pre+list(preds.cpu().numpy())
                # all_target = np.vstack(all_target, target.data.cpu().numpy())
                # all_pre = np.vstack((all_pre,preds.cpu().numpy()))

            # all_target.append(target.data.cpu().numpy())
            # all_pre.append(preds.cpu().numpy())

            #
            output_store_fc.append(pred_score.cpu())
            target_store.append(target.cpu())
            index_vector.append(indexs)

            # if i % args.print_freq == 0:
            #     print('Prec@1 {topframe.val:.3f} ({topframe.avg:.3f})\t'
            #           .format(topframe=topframe))
    np.save('target.npy',np.array(all_target))
    np.save( 'pre.npy',np.array(all_pre))

    index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
    index_matrix = []
    for i in range(int(max(index_vector)) + 1):
        index_matrix.append(index_vector == i)  # 找出相同视频

    index_matrix = torch.stack(index_matrix, dim=0).float()  # [21570]  --->  [380, 21570]

    output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]

    target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]

    pred_matrix_fc = index_matrix.mm(output_store_fc)  # [380,21570] * [21570, 7] = [380,7]
    target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
        index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]

    _, preds = torch.max(pred_matrix_fc.data, 1)
    corrects = float(torch.sum(preds == target_vector.data)) / float(target_vector.size(0))
    topVideo.update(corrects, i + 1)
    print('Test *Prec@Video {topVideo.avg:.3f}   *Prec@Frame {topframe.avg:.3f} '.format(topVideo=topVideo, topframe=topframe))
    print('feature {pred_p1.avg:.3f}   feature_RNN {pred_p2.avg:.3f}   emotion_RNN {pred_p3.avg:.3f} '.format(pred_p1=pred_p1, pred_p2=pred_p2, pred_p3=pred_p3))

    return topVideo.avg,pred_p1.avg,pred_p2.avg,pred_p3.avg


if __name__ == '__main__':

    for i in range(1,11):
        main(i)