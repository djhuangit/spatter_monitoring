#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: DJ
@file: spatter_dataset.py
@time: 2021/06/03
@desc:
"""

import os
import random, csv
import h5py
import numpy as np
import pandas as pd
import visdom

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import cv2


def bytes_decode(io_buf):
    return cv2.imdecode(np.frombuffer(io_buf, np.uint8), -1)


class Spatter(Dataset):

    def __init__(self, file_paths, labels, resize, mode, root=""):
        super(Spatter, self).__init__()

        self.root = root
        self.file_paths = file_paths
        self.labels = labels
        self.resize = resize
        self.data_info = []

        if len(self.file_paths) != len(self.labels):
            print(f"the size of file_paths and labels are not the same. \n\
             len of filepath={len(self.file_paths)} and len of labels={len(self.labels)}")

        self.images, self.labels = self.load_csv("images.csv")

        if mode == 'train':  # 60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]

        elif mode == 'val':  # 20% (60-80%)
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]

        elif mode == 'test':  # 20% (80-100%)
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def __getitem__(self, idx):
        # idx: [0:len(images)]
        # self.images, self.labels
        # img: ['left_low_0.h5', '20210126_074824_section_3_low_0', '0']
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        # print(img)
        file = os.path.join(self.root, img[0])
        ds = img[1]
        ds_idx = int(img[2])

        with h5py.File(file) as h5_file:
            io_buf = h5_file[ds][ds_idx]
            # img = cv2.cvtColor(self.bytes_decode(io_buf), cv2.COLOR_BGR2RGB)
            img = bytes_decode(io_buf)

            tf = transforms.Compose([
                lambda x: Image.fromarray(img).convert('RGB'),  # hdf5 path -> img data
                transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])  # from imagenet
            ])

            img = tf(img)
            label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.images)

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            datasets = []
            ds_lens = []
            ds_labels = []
            ds_files = []

            for file, label in zip(self.file_paths, self.labels):
                # list of dataset: '20210126_074824_section_3_low_0', '20210126_074824_section_3_low_1'
                with h5py.File(os.path.join(self.root, file)) as h5_file:
                    datasets += list(h5_file.keys())

                    for key in list(h5_file.keys()):
                        # list of ds lens: [10000, 10000,..]
                        ds_lens.append(h5_file[key].shape[0])
                        # list of labels: [0, .., 1, .., 2]
                        ds_labels.append(label)
                        # list of file_paths: [left_low_0.h5,left_mid_1.h5, left_high_2.h5]
                        ds_files.append(file)

            # ('left_low_0.h5', '20210126_074824_section_3_low_0', 10000, 0)
            print(list(zip(ds_files, datasets, ds_lens, ds_labels)))

            with open(os.path.join(self.root, 'unshuffled_' + filename), 'w', newline='') as f:
                writer = csv.writer(f)
                # header
                writer.writerow(['file', 'dataset', 'ds_idx', 'label'])
                for idx, ds in enumerate(datasets):
                    for i in range(ds_lens[idx]):
                        # each row: 'file', 'dataset', 'ds_idx', 'label'
                        writer.writerow([ds_files[idx], ds, i, ds_labels[idx]])
                print(f"written into csv file: {os.path.join(self.root, 'unshuffled_' + filename)}")

            df = pd.read_csv(os.path.join(self.root, 'unshuffled_' + filename))
            df = df.sample(frac=1)  # shuffling the entire df
            df.to_csv(os.path.join(self.root, filename), index=False)
            print(f"written into csv file: {os.path.join(self.root, filename)}")

        # read from csv file
        print(f"read csv file: {os.path.join(self.root, filename)}")
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            next(reader)  # skip the csv header line
            for row in reader:
                # each row: 'file', 'dataset', 'ds_idx', 'label'
                ds_file, ds, ds_idx, ds_label = row
                label = int(ds_label)

                images.append([ds_file, ds, ds_idx])
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

class Spatter_gray(Dataset):

    def __init__(self, file_paths, labels, resize, mode, root=""):
        super(Spatter_gray, self).__init__()

        self.root = root
        self.file_paths = file_paths
        self.labels = labels
        self.resize = resize
        self.data_info = []

        if len(self.file_paths) != len(self.labels):
            print(f"the size of file_paths and labels are not the same. \n\
             len of filepath={len(self.file_paths)} and len of labels={len(self.labels)}")

        self.images, self.labels = self.load_csv("images.csv")

        if mode == 'train':  # 60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]

        elif mode == 'val':  # 20% (60-80%)
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]

        elif mode == 'test':  # 20% (80-100%)
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def __getitem__(self, idx):
        # idx: [0:len(images)]
        # self.images, self.labels
        # img: ['left_low_0.h5', '20210126_074824_section_3_low_0', '0']
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        # print(img)
        file = os.path.join(self.root, img[0])
        ds = img[1]
        ds_idx = int(img[2])

        with h5py.File(file) as h5_file:
            io_buf = h5_file[ds][ds_idx]
            # img = cv2.cvtColor(self.bytes_decode(io_buf), cv2.COLOR_BGR2RGB)
            img = bytes_decode(io_buf)

            tf = transforms.Compose([
                lambda x: Image.fromarray(img).convert('L'),  # hdf5 path -> img data
                transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])  # from imagenet
            ])

            img = tf(img)
            label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.images)

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            datasets = []
            ds_lens = []
            ds_labels = []
            ds_files = []

            for file, label in zip(self.file_paths, self.labels):
                # list of dataset: '20210126_074824_section_3_low_0', '20210126_074824_section_3_low_1'
                with h5py.File(os.path.join(self.root, file)) as h5_file:
                    datasets += list(h5_file.keys())

                    for key in list(h5_file.keys()):
                        # list of ds lens: [10000, 10000,..]
                        ds_lens.append(h5_file[key].shape[0])
                        # list of labels: [0, .., 1, .., 2]
                        ds_labels.append(label)
                        # list of file_paths: [left_low_0.h5,left_mid_1.h5, left_high_2.h5]
                        ds_files.append(file)

            # ('left_low_0.h5', '20210126_074824_section_3_low_0', 10000, 0)
            print(list(zip(ds_files, datasets, ds_lens, ds_labels)))

            with open(os.path.join(self.root, 'unshuffled_' + filename), 'w', newline='') as f:
                writer = csv.writer(f)
                # header
                writer.writerow(['file', 'dataset', 'ds_idx', 'label'])
                for idx, ds in enumerate(datasets):
                    for i in range(ds_lens[idx]):
                        # each row: 'file', 'dataset', 'ds_idx', 'label'
                        writer.writerow([ds_files[idx], ds, i, ds_labels[idx]])
                print(f"written into csv file: {os.path.join(self.root, 'unshuffled_' + filename)}")

            df = pd.read_csv(os.path.join(self.root, 'unshuffled_' + filename))
            df = df.sample(frac=1)  # shuffling the entire df
            df.to_csv(os.path.join(self.root, filename), index=False)
            print(f"written into csv file: {os.path.join(self.root, filename)}")

        # read from csv file
        print(f"read csv file: {os.path.join(self.root, filename)}")
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            next(reader)  # skip the csv header line
            for row in reader:
                # each row: 'file', 'dataset', 'ds_idx', 'label'
                ds_file, ds, ds_idx, ds_label = row
                label = int(ds_label)

                images.append([ds_file, ds, ds_idx])
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

class Spatter_gray_ratio(Dataset):
    '''
    Require images.csv that contains sorted (0 to 1) and perfectly balanced dataset
    '''
    def __init__(self, file_paths, labels, resize, cls1_ratio, root=""):
        super(Spatter_gray_ratio, self).__init__()

        self.root = root
        self.file_paths = file_paths
        self.labels = labels
        self.resize = resize
        self.data_info = []

        if len(self.file_paths) != len(self.labels):
            print(f"the size of file_paths and labels are not the same. \n\
             len of filepath={len(self.file_paths)} and len of labels={len(self.labels)}")

        self.images, self.labels = self.load_csv("images.csv")

        if cls1_ratio >= 0.5:
            self.images = self.images[int((0.5 - (1-cls1_ratio)/cls1_ratio/2) * len(self.images)):]
            self.labels = self.labels[int((0.5 - (1-cls1_ratio)/cls1_ratio/2) * len(self.labels)):]
        else:
            self.images = self.images[:int((0.5 + cls1_ratio/(1-cls1_ratio)/2) * len(self.images))]
            self.labels = self.labels[:int((0.5 + cls1_ratio/(1-cls1_ratio)/2) * len(self.labels))]

    def __getitem__(self, idx):
        # idx: [0:len(images)]
        # self.images, self.labels
        # img: ['left_low_0.h5', '20210126_074824_section_3_low_0', '0']
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        # print(img)
        file = os.path.join(self.root, img[0])
        ds = img[1]
        ds_idx = int(img[2])

        with h5py.File(file) as h5_file:
            io_buf = h5_file[ds][ds_idx]
            # img = cv2.cvtColor(self.bytes_decode(io_buf), cv2.COLOR_BGR2RGB)
            img = bytes_decode(io_buf)

            tf = transforms.Compose([
                lambda x: Image.fromarray(img).convert('L'),  # hdf5 path -> img data
                transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])  # from imagenet
            ])

            img = tf(img)
            label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.images)

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            datasets = []
            ds_lens = []
            ds_labels = []
            ds_files = []

            for file, label in zip(self.file_paths, self.labels):
                # list of dataset: '20210126_074824_section_3_low_0', '20210126_074824_section_3_low_1'
                with h5py.File(os.path.join(self.root, file)) as h5_file:
                    datasets += list(h5_file.keys())

                    for key in list(h5_file.keys()):
                        # list of ds lens: [10000, 10000,..]
                        ds_lens.append(h5_file[key].shape[0])
                        # list of labels: [0, .., 1, .., 2]
                        ds_labels.append(label)
                        # list of file_paths: [left_low_0.h5,left_mid_1.h5, left_high_2.h5]
                        ds_files.append(file)

            # ('left_low_0.h5', '20210126_074824_section_3_low_0', 10000, 0)
            print(list(zip(ds_files, datasets, ds_lens, ds_labels)))

            with open(os.path.join(self.root, 'unshuffled_' + filename), 'w', newline='') as f:
                writer = csv.writer(f)
                # header
                writer.writerow(['file', 'dataset', 'ds_idx', 'label'])
                for idx, ds in enumerate(datasets):
                    for i in range(ds_lens[idx]):
                        # each row: 'file', 'dataset', 'ds_idx', 'label'
                        writer.writerow([ds_files[idx], ds, i, ds_labels[idx]])
                print(f"written into csv file: {os.path.join(self.root, 'unshuffled_' + filename)}")

            df = pd.read_csv(os.path.join(self.root, 'unshuffled_' + filename))
            df = df.sample(frac=1)  # shuffling the entire df
            df.to_csv(os.path.join(self.root, filename), index=False)
            print(f"written into csv file: {os.path.join(self.root, filename)}")

        # read from csv file
        print(f"read csv file: {os.path.join(self.root, filename)}")
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            next(reader)  # skip the csv header line
            for row in reader:
                # each row: 'file', 'dataset', 'ds_idx', 'label'
                ds_file, ds, ds_idx, ds_label = row
                label = int(ds_label)

                images.append([ds_file, ds, ds_idx])
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

def main():
    import visdom
    import time

    viz = visdom.Visdom()

    # using self implemented Dataset class

    # 9 files, 3 labels
    # file_list = ['left_low_0.h5', 'left_mid_1.h5', 'left_high_2.h5',
    #              'mid_low_0.h5', 'mid_mid_1.h5', 'mid_high_2.h5',
    #              'right_low_0.h5', 'right_mid_1.h5', 'right_high_2.h5']
    # label_list = [0, 1, 2, 0, 1, 2, 0, 1, 2] # 3 classes={0:low gas flow, 1: mid gas flow, 2: high gas flow}

    # 6 files, 2 labels
    file_list = ['left_low_0.h5', 'left_high_2.h5',
                 'mid_low_0.h5', 'mid_high_2.h5',
                 'right_low_0.h5', 'right_high_2.h5']
    label_list = [0, 1, 0, 1, 0, 1]  # 2 classes={0:low gas flow, 1: high gas flow}

    db = Spatter(file_list, label_list, resize=224)
    # print(db.images[0], db.labels[0])

    print(f"total size: {db.__len__()}")
    # db.__getitem__(10)
    x, y = next(iter(db))
    print(x.shape, y)

    viz.image(x, win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True)

    for x, y in loader:
        viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy().reshape(4, 8)), win='label', opts=dict(title='batch_y'))

        time.sleep(10)


if __name__ == '__main__':
    main()
