from __future__ import print_function
import torch
import torch.utils.data
import torchvision.transforms as transforms
from Facial.Model import class_data_new1

cate2label = {'CK+':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6},

              'ESVG': {0: 'An', 1: 'Di', 2: 'Fe', 3: 'Ha', 4: 'Ne', 5: 'Sa', 6: 'Su',
                       'An': 0, 'Di': 1, 'Fe': 2, 'Ha': 3, 'Ne': 4, 'Sa': 5, 'Su': 6},

              'Enterface': {0: 'an', 1: 'di', 2: 'fe', 3: 'ha', 4: 'sa', 5: 'su',
                       'an': 0, 'di': 1, 'fe': 2, 'ha': 3, 'sa': 4, 'su': 5},

              }


cate2label = cate2label['CK+']

def LoadData(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval,C,K):

    train_dataset = class_data_new1.VideoDataset3(
        video_root=root_train,
        video_list=list_train,
        C=C,K=K,
        rectify_label=cate2label,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]),
    )

    val_dataset = class_data_new1.VideoDataset3(
        video_root=root_eval,
        video_list=list_eval,
        C=C, K=K,
        rectify_label=cate2label,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]),
    )

    # test_dataset = class_data_new1.VideoDataset1(
    #     video_root=root_eval,
    #     video_list=list_eval,
    #     C=C, K=14,
    #     rectify_label=cate2label,
    #     transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    # )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True)#, drop_last=True

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False)

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1, shuffle=False)

    return train_loader, val_loader#,test_loader

if __name__ == '__main__':
    dir_train = 'E:/CK+/video'
    list_train = '../Data/test.txt'

    dir_eval = 'E:/CK+/video'
    list_eval = '../Data/test.txt'

    C=7
    K=3
    train_loader, val_loader = LoadData(dir_train, list_train, 4, dir_eval, list_eval, 4, C, K)

    for i, (video, index) in enumerate(train_loader):

        target = video[0][1]  # 类别标签
        input = video[0][0].unsqueeze(0)  # frame 【1，batch，length，channel，】
        indexs = index  # 视频编号
        for j in range(1, C):
            # target = torch.cat([target, video[j][1]], 0)
            input = torch.cat([input, video[j][0].unsqueeze(0)], 0)
            # indexs= torch.cat([indexs,index],0)

        # input 【clip_length，batch，frame_length，channel
        target = target.cuda()
        input = input.view(-1, K, 3, 224, 224).permute(0, 2, 1, 3, 4).cuda()

        #
        # target2 = torch.stack([video[0][1], video[1][1], video[2][1], video[3][1], video[4][1]]).view(-1).cuda()
        # target=video[0][1]
        # input=video[0][0].unsqueeze(0)
        # for j in range(1,C):
        #     target=torch.cat([target,video[j][1]],0)
        #     input=torch.cat([input,video[j][0].unsqueeze(0)],0)
        # input = input.view(-1, K, 3, 224, 224).permute(0,2,1,3,4).cuda()
        print(1)
