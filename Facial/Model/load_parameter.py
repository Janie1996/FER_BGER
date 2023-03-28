from __future__ import print_function
import os
import torch
import torch.utils.data


def LoadParameter(_structure, _parameterDir):
    """

    :param _structure: model
    :param _parameterDir: 参数位置
    :return:
    """
    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if key in model_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
    """
    another method:
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
    model_state_dict.update(pretrained_dict)
    """

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model


def savecheckpoint(state):
    """
    :param state: 预保存参数信息
    e.g  savecheckpoint({
                'epoch': 1,
                'state_dict': model.state_dict(),  # Out,in,kernel_size,kernel_size
                'prec1': 0,
            })
    :return:
    """
    if not os.path.exists('./model'):
        os.makedirs('./model')
    epoch = state['epoch']
    save_dir = './model/' + str(epoch) + '_' + str(round(float(state['prec1']), 4))
    torch.save(state, save_dir)
    #print(save_dir)



def savecheckpoint(state,dir):
    """
    :param state: 预保存参数信息
    e.g  savecheckpoint({
                'epoch': 1,
                'state_dict': model.state_dict(),  # Out,in,kernel_size,kernel_size
                'prec1': 0,
            })
    :return:
    """
    save_dir = dir + '_' + str(round(float(state['prec1']), 4))
    torch.save(state, save_dir)