# -*- coding: utf-8 -*-

class Config:
    # for data_process.py
    root = r'/data1/jiawenxiao/physionet2020_official/code/'
    # for train
    #训练的模型名称
    model_name = 'resnet34'
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [10,12]
    #训练时的batch大小
    batch_size = 64
    #label的类别数
    num_classes = 27
    #最大训练多少个epoch
    max_epoch = 14
    #目标的采样长度
    target_point_num = 5000
    #保存模型的文件夹
    ckpt = 'ckpt'
    #保存提交文件的文件夹
    sub_dir = 'submit'
    #初始的学习率
    lr = 1e-3
    #保存模型当前epoch的权重
    current_w = 'current_w.pth'
    #保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10
config = Config()
