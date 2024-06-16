import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class WikiArtDataset(Dataset):

    def __init__(self, root_dir, category, transform=None, size=None, label_filters=None):
        if category == 'artist' or category == 'genre' or category == 'style':
            self.category = category
        else:
            raise ValueError('category must be either "artist" or "genre" or "style"')

        self.category = category
        self.root_dir = root_dir
        self.transform = transform

        class_file = os.path.join(root_dir, f'{category}_class.txt')
        labels = pd.read_csv(class_file, delimiter=' ', header=None)
        self.label_dict = labels[1]
        print(f'Category labels\': \n{self.label_dict}')

        category_train = os.path.join(root_dir, f'{category}_train.csv')
        print(f'Load images from : {category_train}')
        columns = ['Image_Path', 'Label']
        self.train_list = pd.read_csv(category_train, header=None, names=columns)
        if size is not None:
            self.train_list = self.train_list[:size]

        if label_filters is not None:
            selected_labels = [self.label_dict[key] for key in label_filters]
            print(f'WikiArt Label filters: {selected_labels}')
            self.train_list = self.train_list.loc[self.train_list['Label'].isin(label_filters)]
            self.train_list = self.train_list.reset_index(drop=True)

        print(f'Dataset size: {len(self.train_list)}')

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.train_list.Image_Path[idx])
        try:
            image = io.imread(img_name)
        except IOError:
            print(f"Warning: Skipping corrupted image {img_name}")
            return None  # or use a default image

        if self.transform:
            image = self.transform(image)

        label = self.train_list.Label[idx]
        return (image, self.label_dict[label])
