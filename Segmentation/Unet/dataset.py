# *_*coding:utf-8 *_*
# @Author : yuemengrui
# @Time : 2021-06-01 下午3:46
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms


class TableSegDataset(Dataset):

    def __init__(self, data_dir, mode='train', target_size=(608, 608), batch_size=2):
        assert mode in ['train', 'val']
        self.mode = mode
        self.target_size = target_size
        self.img_list = []
        self.label_list = []
        if batch_size > 1:
            self.img_32n = False
        else:
            self.img_32n = True

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])
        if self.mode == 'train':
            train_label_path = os.path.join(data_dir, 'train.txt')
            with open(train_label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                img_path = os.path.join(data_dir, line.split(' ')[0])
                label_path = os.path.join(data_dir, line.split(' ')[1].strip())
                self.img_list.append(img_path)
                self.label_list.append(label_path)
        else:
            val_label_path = os.path.join(data_dir, 'val.txt')
            with open(val_label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                img_path = os.path.join(data_dir, line.split(' ')[0])
                label_path = os.path.join(data_dir, line.split(' ')[1].strip())
                self.img_list.append(img_path)
                self.label_list.append(label_path)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        label_img = np.uint8(np.zeros((h, w)))
        label_img = self.__create_label_img(label_path, label_img)

        if self.img_32n:
            h_scale = int(h // 32) if int(h // 32) > 2 else 2
            w_scale = int(w // 32) if int(w // 32) > 2 else 2
            self.target_size = (32 * w_scale, 32 * h_scale)

        img = cv2.resize(img, self.target_size)
        label_img = cv2.resize(label_img, self.target_size, interpolation=cv2.INTER_NEAREST)
        img = self.transform(img)
        return img, label_img.astype(np.int64)
        # return img, label_img

    def __len__(self):
        return len(self.label_list)

    def __create_label_img(self, label_path, label_img):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        rows = []
        cols = []
        un_rows = []
        un_cols = []
        for line in lines:
            line_splits = line.split(',')
            if line_splits[-1].strip() == 'row':
                rows.append([int(line_splits[0]), int(line_splits[1]), int(line_splits[2]), int(line_splits[3])])
            elif line_splits[-1].strip() == 'col':
                cols.append([int(line_splits[0]), int(line_splits[1]), int(line_splits[2]), int(line_splits[3])])
            elif line_splits[-1].strip() == 'un_row':
                un_rows.append([int(line_splits[0]), int(line_splits[1]), int(line_splits[2]), int(line_splits[3])])
            elif line_splits[-1].strip() == 'un_col':
                un_cols.append([int(line_splits[0]), int(line_splits[1]), int(line_splits[2]), int(line_splits[3])])
            else:
                raise ValueError(
                    "label error!!! {}.{}".format(line_splits[-1].strip(), label_path))

        if random.random() > 0.5:
            for un_r in un_rows:
                label_img[un_r[1]:un_r[3], un_r[0]:un_r[2]] = 1

            for un_c in un_cols:
                label_img[un_c[1]:un_c[3], un_c[0]:un_c[2]] = 2

            for r in rows:
                label_img[r[1]:r[3], r[0]:r[2]] = 3

            for c in cols:
                label_img[c[1]:c[3], c[0]:c[2]] = 4
        else:
            for un_c in un_cols:
                label_img[un_c[1]:un_c[3], un_c[0]:un_c[2]] = 2

            for un_r in un_rows:
                label_img[un_r[1]:un_r[3], un_r[0]:un_r[2]] = 1

            for c in cols:
                label_img[c[1]:c[3], c[0]:c[2]] = 4

            for r in rows:
                label_img[r[1]:r[3], r[0]:r[2]] = 3

        return label_img


if __name__ == '__main__':
    dataset = TableSegDataset(
        data_dir='/Users/yuemengrui/MyWork/Table_Segmentation/dataset_handle/table_segmentation_dataset', mode='val', batch_size=1)

    # train_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    for img, label in dataset:
        # print(img.shape)
        # print(label.shape)
        # print("===============================================")
        # cv2.imshow('xx', img)
        # cv2.waitKey(0)
        # cv2.imshow('label', label)
        # cv2.waitKey(0)
        # im = img.copy()
        # im[label == 0] = [0, 0, 255]
        # im[label == 1] = [0, 0, 0]
        # im[label == 2] = [255, 0, 255]
        # im[label == 3] = [0, 255, 255]
        # im[label == 4] = [255, 0, 0]
        # show = cv2.addWeighted(img, 0.6, im, 0.4, 0)
        # cv2.imshow('show', show)
        # cv2.waitKey(0)
        label = label.reshape(-1)
        print(np.bincount(label))
