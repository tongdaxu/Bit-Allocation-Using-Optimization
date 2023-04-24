import os
import torch.nn as nn
import torch
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
import torch.nn.functional as F
from torchvision.utils import save_image


class HEVCDataSet(data.Dataset):
    def __init__(self, class_, gop_size, test_gop_num):
        root = "xx" + class_
        filelist = "./Tools/filelists/" + class_ + ".txt"
        with open(filelist) as f:
            folders = f.readlines()
        self.folders = folders
        self.input = []
        self.hevcclass = []
        for folder in folders:
            seq = folder.rstrip()
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
                if cnt == gop_size * test_gop_num:
                    break
            for i in range(test_gop_num):
                inputpath = []
                for j in range(gop_size):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * gop_size + j + 1).zfill(3) + '.png'))
                self.input.append(inputpath)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_images = []
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0
            h = int((input_image.shape[1] // 64) * 64)
            w = int((input_image.shape[2] // 64) * 64)
            input_images.append(input_image[:, :h, :w])
        input_images = np.array(input_images)

        return input_images, self.folders


class UVGDataSet(data.Dataset):
    def __init__(self, gop_size, test_gop_num):
        root = "xx"
        filelist = "./Tools/filelists/UVG.txt"
        with open(filelist) as f:
            folders = f.readlines()
        self.input = []
        self.hevcclass = []
        for folder in folders:
            seq = folder.rstrip()
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
                if cnt == gop_size * test_gop_num:
                    break
            for i in range(test_gop_num):
                inputpath = []
                for j in range(gop_size):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * gop_size + j + 1).zfill(3) + '.png'))
                self.input.append(inputpath)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_images = []
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0
            h = int((input_image.shape[1] // 64) * 64)
            w = int((input_image.shape[2] // 64) * 64)
            input_images.append(input_image[:, :h, :w])
        input_images = np.array(input_images)
        return input_images


def crop(input_image, factor=4, overlap=0):
    """
    :param input_image: input image to be cropped.
    :param factor: use this arg to control the biggest height and width of blocks, default: 4.
    :param overlap: the size of overlapping area, default: 0.
    :return: cropped image stack and the initial size of un-cropped image.
    """
    B, C, H, W = input_image.shape
    assert (H % 64 == 0 and W % 64 == 0), "H and W must be a multiple of 64"
    assert (overlap == 0 or 64)
    maxsize = [64 * factor, 64 * factor]
    h_num, w_num = math.ceil(H / maxsize[0]), math.ceil(W / maxsize[1])
    cropped_image_stack = []
    for h in range(h_num):
        for w in range(w_num):
            h_down = 0 if h * maxsize[0] - overlap < 0 else h * maxsize[0] - overlap
            h_up = H if (h + 1) * maxsize[0] + overlap > H else (h + 1) * maxsize[0] + overlap
            w_down = 0 if w * maxsize[1] - overlap < 0 else w * maxsize[1] - overlap
            w_up = W if (w + 1) * maxsize[1] + overlap > W else (w + 1) * maxsize[1] + overlap

            cropped_image_stack.append(input_image[:, :, h_down: h_up, w_down: w_up])
    return cropped_image_stack, B, C, H, W


def merge(cropped_image_stack, B, C, H, W, factor=4, overlap=0):
    """
    :param cropped_image_stack: the images to be merged.
    :param B, C, H, W: the initial size of un-cropped image.
    :param factor: use this arg to control the biggest height and width of blocks, default: 4.
    :param overlap: the size of overlapping area, default: 0.
    :return: merged image.
    """
    assert (H % 64 == 0 and W % 64 == 0), "H and W must be a multiple of 64"
    assert (overlap == 0 or 64)
    maxsize = [64 * factor, 64 * factor]
    h_num, w_num = math.ceil(H / maxsize[0]), math.ceil(W / maxsize[1])
    merged_image = torch.zeros([B, C, H, W])
    i = 0
    for h in range(h_num):
        for w in range(w_num):
            h_down = 0 if h == 0 else overlap
            h_up = H if h + 1 == h_num else cropped_image_stack[i].shape[2] - overlap
            w_down = 0 if w == 0 else overlap
            w_up = W if w + 1 == w_num else cropped_image_stack[i].shape[3] - overlap

            h_pad_down = h * maxsize[0]
            h_pad_up = H if (h + 1) * maxsize[0] > H else (h + 1) * maxsize[0]
            w_pad_down = w * maxsize[1]
            w_pad_up = W if (w + 1) * maxsize[1] > W else (w + 1) * maxsize[1]

            merged_image[:, :, h_pad_down: h_pad_up, w_pad_down: w_pad_up] = cropped_image_stack[i][:, :, h_down: h_up, w_down: w_up]
            i += 1
    return merged_image


def get_pad_result(cur_frame):
    h = int(128 - cur_frame.shape[2] % 128)
    w = int(128 - cur_frame.shape[3] % 128)

    if h % 128 == 0 and w % 128 == 0:  # no padding
        return cur_frame
    else:
        cur_frame = F.pad(cur_frame, (0, w, 0, h), 'constant', 0)  # left = 0, right = w, up = 0, down = h
        return cur_frame

