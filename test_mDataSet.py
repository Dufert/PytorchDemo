# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:33:20 2019

@author: Administrator
"""
import matplotlib.pyplot as plt
import torch.utils.data as data
from mDataSet import mDataSet

mds = mDataSet('h:/Library/FaceDataHub/mylibrary/train/')
mDataLoader = data.DataLoader(mds,batch_size=5,shuffle=True)

for step, (bx, by) in enumerate(mDataLoader):
    a = bx.numpy()
    b = by.numpy()
    print(a.shape, b)    
    
    
    