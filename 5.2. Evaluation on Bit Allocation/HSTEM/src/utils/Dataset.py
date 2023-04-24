import os
import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset


class VimeoDataset(Dataset):
    def __init__(self, h5_file):
        super(VimeoDataset, self).__init__()
        self.h5 = h5py.File(name=h5_file, mode='r')
        self.file_list = list(self.h5.keys())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):  # 负责按索引取出某个数据，并对该数据做预处理
        frames = self.h5[self.file_list[idx]]
        frames = np.array(frames)
        frames = torch.from_numpy(frames) / 255.0  # scale to [0, 1]
        frames = frames.permute(0, 3, 1, 2).contiguous().float()
        return frames


def is_h5(h5):
    return any(h5.endswith(extension) for extension in ['.hdf5', ])


def create_dataset(h5_folder):
    datasets = [VimeoDataset(os.path.join(h5_folder, h5)) for h5 in os.listdir(h5_folder) if is_h5(h5)]
    return ConcatDataset(datasets=datasets)

#
# tester = VimeoDataset('../Dataset/Train_1024_part_9.hdf5')
# from PIL import Image
#
# aa = tester[2]
#
# for i in range(7):
#     a = aa[i]
#
#     a = a.permute(1, 2, 0).contiguous()
#     a = a.numpy() * 255
#     p = a.astype(np.uint8)
#     p = Image.fromarray(p)
#     p.show()
#     input()
#
# tester = create_dataset('../Dataset/Eval')
# print(len(tester))
# from PIL import Image
#
# aa = tester[90]
#
# for i in range(7):
#     a = aa[i]
#
#     a = a.permute(1, 2, 0).contiguous()
#     a = a.numpy() * 255
#     p = a.astype(np.uint8)
#     p = Image.fromarray(p)
#     p.show()
#     input()
