# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset
from model import Model

dataset = Dataset()
model = Model(dataset)

# 调用predict_all的方法
x_test = [{'image_path': 'images/2711.jpg'}, {'image_path': 'images/240.jpg'},
          {'image_path': 'images/27.jpg'}, {'image_path': 'images/28.jpg'}]
y_test = [{'label': 2}, {'label': 0},
          {'label': 0}, {'label': 0}]
preds = model.predict_all(x_test)
labels = [i['label'] for i in y_test]
print(preds)
# 调用predict的方法
img_path = 'images/240.jpg'
p = model.predict(image_path=img_path)
print(p)
