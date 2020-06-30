# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017
@author: user
这是一个空的样例代码
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flyai.dataset import Dataset
from porn_dataset import PornDataset
from model import Model
from losses import CELoss
from pornnet import PornNet
from path import MODEL_PATH

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''


def train():
    lowest_loss = 100
    dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
    model = Model(dataset)

    device = torch.device('cuda')
    net = PornNet('litnet')
    loss = CELoss()
    net.to(device)
    loss.to(device)
    #
    optimizer = torch.optim.Adam(net.parameters(), 5.0e-4, weight_decay=1.0e-4)
    # 获取所有原始数据
    x_train, y_train, x_val, y_val = dataset.get_all_data()

    # 构建自己的数据加载器
    train_dataset = PornDataset(x_train, y_train)
    valid_dataset = PornDataset(x_val, y_val)

    train_batch_size = args.BATCH
    valid_batch_size = 8
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=train_batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True)
    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=valid_batch_size,
                                   shuffle=False,
                                   num_workers=0)

    for epoch in range(args.EPOCHS):
        net, train_loss = train_epoch(net, loss, train_data_loader, optimizer, device)
        val_loss = val_epoch(net, loss, valid_data_loader, device)
        print('epoch: %d, train_loss: %f, val_loss: %f' % (epoch + 1, train_loss, val_loss))
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            # 保存模型
            print("lowest loss: %f" % lowest_loss)
    model.save_model(net, MODEL_PATH)


def train_epoch(model, loss, data_loader, optimizer, device):
    model.train()
    loss.train()
    epoch_loss = 0.0
    for batch_item in data_loader:
        batch_img, batch_label = batch_item
        # print(batch_img.shape)
        batch_img = batch_img.to(device=device)

        batch_label = batch_label.to(device=device)

        batch_prediction = model(batch_img)
        batch_loss = loss(batch_prediction, batch_label)
        # import pdb
        # pdb.set_trace()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss

    epoch_loss = epoch_loss  # / len(data_loader)

    return model, epoch_loss


def val_epoch(model, loss, data_loader, device):
    model.eval()
    loss.eval()
    epoch_loss = 0.0
    for batch_item in data_loader:
        batch_img, batch_label = batch_item
        batch_img = batch_img.to(device=device)

        batch_label = batch_label.to(device=device)

        batch_prediction = model(batch_img)
        batch_loss = loss(batch_prediction, batch_label)
        epoch_loss += batch_loss

    epoch_loss = epoch_loss  # / len(data_loader)
    return epoch_loss


if __name__ == '__main__':
    train()
