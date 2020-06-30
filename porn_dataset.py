import cv2
import os
import torchvision
import torch.utils.data as data
import random
from path import DATA_PATH
import matplotlib.pyplot as plt


# 自定义获取数据的方式
class PornDataset(data.Dataset):
    def __init__(self, img_path_list, label_list):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_PATH, self.img_path_list[idx]['image_path'])
        label = self.label_list[idx]['label']
        img = cv2.imread(img_path)
        res_img = process_img(img)
        # plt.imshow(res_img)
        # plt.show()
        return self.transform(res_img), label

    def __len__(self):
        return len(self.img_path_list)


def process_img(img):
    h, w, _ = img.shape
    # plt.imshow(img)
    # plt.show()

    if h < w:
        rate = h / 512
        res_h, res_w = (512, int(w / rate))
        new_img = cv2.resize(img, (res_w, res_h))
        if res_w - 512 > 4:
            step = (res_w - 512) // 4
            random_list = (0, step - 1, 2 * step - 1, 3 * step - 1, 4 * step - 1)
            random_start = random.choice(random_list)
            res_img = new_img[:, random_start:random_start + 512, :]
        else:
            res_img = new_img[:, 0:512, :]
    else:
        rate = w / 512
        res_h, res_w = (int(h / rate), 512)
        new_img = cv2.resize(img, (res_w, res_h))
        if res_h - 512 > 4:
            step = (res_h - 512) // 4
            random_list = (0, step - 1, 2 * step - 1, 3 * step - 1, 4 * step - 1)
            random_start = random.choice(random_list)
            res_img = new_img[random_start:random_start + 512, :, :]
        else:
            res_img = new_img[0:512, :, :]
    return res_img

