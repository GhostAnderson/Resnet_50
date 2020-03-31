import torch
import numpy as np
import random
from PIL import Image

class Dataset(torch.utils.data.Dataset):

    def __init__(self,path,transform = None,train = True):
        self.transform = transform
        self.img_set = []
        self.path = path
        print("path:", path)
        import os
        if train:
            birds_path = os.path.join(path,'train/birds')
            print(birds_path)
            birds_set = list(map(lambda x:os.path.join(birds_path,x),os.listdir(birds_path)))
            list(map(lambda x:self.img_set.append((x,1)),birds_set))

            nonbirds_path = os.path.join(path,'train/nonbirds')
            nonbirds_set = list(map(lambda x:os.path.join(nonbirds_path,x),os.listdir(nonbirds_path)))
            list(map(lambda x:self.img_set.append((x,0)),nonbirds_set))
        else:
            birds_path = os.path.join(path,'test/birds')
            birds_set = list(map(lambda x:os.path.join(birds_path,x),os.listdir(birds_path)))
            list(map(lambda x:self.img_set.append((x,1)),birds_set))

            nonbirds_path = os.path.join(path,'test/nonbirds')
            nonbirds_set = list(map(lambda x:os.path.join(nonbirds_path,x),os.listdir(nonbirds_path)))
            list(map(lambda x:self.img_set.append((x,0)),nonbirds_set))

    def __getitem__(self, index):
        path,label = self.img_set[index]
        if not self.transform == None:
            return self.transform(Image.open(path)),label
        else:
            return Image.open(path),label

    def __len__(self):
        return len(self.img_set)