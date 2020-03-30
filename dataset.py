import torch
import numpy as np
import random

class Dataset(torch.utils.data.Dataset):

    def __init__(self,transform = None,train = True):
        self.transform = transform
        self.img_set = None
        self.label_set = None
        if train:
            self.img_set = 1 #TODO: 载入训练集
            self.label_set = 1
        else:
            self.img_set = 0 #TODO: 载入测试集
            self.label_set = 0

    def __getitem__(self, index):
        a = np.random([1,3,224,224]) #TODO: 从self.img_set中载入
        b = random.randint(0,1)      #TODO: 从self.label_set中载入
        if not self.transform == None:
            return self.transform(a),b
        else:
            return a,b

    def __len__(self):
        return 10