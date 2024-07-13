import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

corrupted_images = {}

class WikiArtDataset(Dataset):

    def __init__(self, root_dir, category, transform=None, size=None, label_filters=None):
        global corrupted_images  # Ensure we're using the global variable
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

        # Train
        category_train = os.path.join(root_dir, f'{category}_train.csv')
        print(f'Load images from : {category_train}')
        columns = ['Image_Path', 'Label']
        train_list = pd.read_csv(category_train, header=None, names=columns)
        if size is not None:
            train_list = train_list[:size]

        # Test
        category_train = os.path.join(root_dir, f'{category}_val.csv')
        print(f'Load images from : {category_train}')
        columns = ['Image_Path', 'Label']
        test_list = pd.read_csv(category_train, header=None, names=columns)
        if size is not None:
            test_list = test_list[:size]

        # Merge train and test lists
        self.data_list = pd.concat([train_list, test_list], ignore_index=True)

        if label_filters is not None:
            selected_labels = [self.label_dict[key] for key in label_filters]
            print(f'WikiArt Label filters: {selected_labels}')
            self.data_list = self.data_list.loc[self.data_list['Label'].isin(label_filters)]
            self.data_list = self.data_list.reset_index(drop=True)

        print(f'Dataset size: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        global corrupted_images  # Ensure we're using the global variable
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_list.Image_Path[idx])
        try:
            image = io.imread(img_name)
        except IOError:
            if idx not in corrupted_images:
                # print(f"Warning: Skipping corrupted image {img_name}")
                corrupted_images[idx] = img_name
            return None  # or use a default image

        if self.transform:
            image = self.transform(image)

        label = self.data_list.Label[idx]
        return (image, self.label_dict[label])
