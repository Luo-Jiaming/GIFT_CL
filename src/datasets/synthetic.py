import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import pickle

from torch.utils.data import Dataset

from .imagenet_classnames import get_classnames
from ..templates.openai_imagenet_template import openai_imagenet_template

class SyntheticDataset(Dataset):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=64,
                 batch_size_eval=64,
                 num_workers=32,
                 image_nums=1000,
                 shuffle=True):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.eval_batch_size = batch_size_eval
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.image_files = [f for f in os.listdir(self.location) if f.endswith('.jpeg') and not self.is_black(os.path.join(self.location, f))]
    
        self.image_files = self.image_files[:image_nums]

        df = pd.read_csv(self.location + 'image_text.csv') # for now
        self.prompt_dict = df.set_index('image')['text'].to_dict()
        self.all_prompts = [self.prompt_dict[os.path.join(self.location,image_file)] for image_file in self.image_files]

        self.all_prompts = list(set(self.all_prompts))
        self.label_dict = {}

        for image in self.image_files:
            self.label_dict[os.path.join(self.location, image)] = self.all_prompts.index(self.prompt_dict[os.path.join(self.location, image)])

        self.populate_train()
        self.populate_test()
    
    def is_black(self, image_path):
        image = Image.open(image_path).convert('RGB')
        array = np.array(image)
        return np.all(array == 0)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.location, image_file)
        image = Image.open(image_path).convert('RGB')

        # add other transform here

        if self.preprocess:
            image = self.preprocess(image)

        return {
            'images': image,
            'labels': self.label_dict[image_path],
            'texts': self.prompt_dict[image_path]
        }

    def populate_train(self):
        self.train_loader = torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle
        )
    
    def populate_test(self):
        self.test_loader = torch.utils.data.DataLoader(
            self,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle
        )