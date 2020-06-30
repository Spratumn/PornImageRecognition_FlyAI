# -*- coding: utf-8 -*
import numpy as np
import os
import torch
from flyai.model.base import Base
import cv2
from path import MODEL_PATH, DATA_PATH
from porn_dataset import process_img
import torchvision
__import__('net', fromlist=["Net"])

from collections import Counter
TORCH_MODEL_NAME = "model.pkl"


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        if os.path.exists(self.net_path):
            self.net = torch.load(self.net_path)

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)

        # x_data = self.data.predict_data(**data)
        # x_data = torch.from_numpy(x_data)
        image_name = data['image_path']
        x_data = cv2.imread(os.path.join(DATA_PATH, image_name))
        pred_list = []
        for i in range(7):
            res_img = process_img(x_data)
            res_img = torchvision.transforms.ToTensor()(res_img).unsqueeze(0).to(torch.device('cuda'))
            outputs = self.net(res_img)[0]
            prediction = outputs.data.cpu().numpy()
            prediction = self.data.to_categorys(prediction)
            pred_list.append(prediction)

        pred_counts = Counter(pred_list)
        res_pred = pred_counts.most_common(1)[0][0]
        return res_pred

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.net_path)
        labels = []
        for data in datas:
            image_name = data['image_path']
            x_data = cv2.imread(os.path.join(DATA_PATH, image_name))
            pred_list = []
            for i in range(7):
                res_img = process_img(x_data)
                res_img = torchvision.transforms.ToTensor()(res_img).unsqueeze(0).to(torch.device('cuda'))
                outputs = self.net(res_img)[0]
                prediction = outputs.data.cpu().numpy()
                prediction = self.data.to_categorys(prediction)
                pred_list.append(prediction)

            pred_counts = Counter(pred_list)
            res_pred = pred_counts.most_common(1)[0][0]
            labels.append(res_pred)
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))


