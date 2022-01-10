import torch
import os
import numpy as np
import math
import h5py
import cv2
import matplotlib.pyplot as plt


def cut66_200(image):
    return image[0:200][11:77]


def getfilenamelist(path):
    FNlist = []
    for root, dirs, files in os.walk(path):
        for f in files:
            FNlist.append(os.path.join(root, f))
    FNlist.sort()
    return FNlist


def BrightnessHistogramEqualization(originimage):
    imageYUV = cv2.cvtColor(originimage, cv2.COLOR_BGR2YCrCb)
    channelsYUV = cv2.split(imageYUV)
    channelsYUV[0] = cv2.equalizeHist(channelsYUV[0])
    imageYUV = cv2.merge(channelsYUV)
    resultimage = cv2.cvtColor(imageYUV, cv2.COLOR_YCrCb2BGR)
    return resultimage


def BHE(originimage):
    return BrightnessHistogramEqualization(originimage)


def ImageNormalization(image):
    if np.max(image) - np.min(image) == 0:
        plt.title('Image 1')
        plt.imshow(image)
        plt.show()
    assert (np.max(image) - np.min(image) != 0)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def IN(image):
    return ImageNormalization(image)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filenamelist):
        self.filenamelist = filenamelist

    def __len__(self):
        return int(len(self.filenamelist)*200)

    def __getitem__(self,idx):
        file_idx = math.floor(idx / 200)
        infile_idx = math.floor(idx % 200)
        filename = self.filenamelist[file_idx]
        readfile = h5py.File(filename, 'r')
        image = readfile["images"][infile_idx]
        target = readfile["targets"][infile_idx]
        steer = target[0]
        assert (np.max(image) - np.min(image) != 0)
        image = cut66_200(image)
        image = BHE(image)
        image = IN(image)
        readfile.close()
        return image, np.array([steer])
