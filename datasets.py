import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import v2

class MyDataSet(data.Dataset):
    def __init__(self, age_min, age_max, image_dir, label_dir, output_size=(256, 256), training_set=True, obscure_age=True):
        self.image_dir = image_dir
        self.transform = v2.Compose([
            v2.Resize(output_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common normalization for pre-trained models
        ])
        # Load label file
        label = np.load(label_dir)
        train_len = int(0.95 * len(label))
        self.training_set = training_set
        self.obscure_age = obscure_age
        if training_set:
            label = label[:train_len]
        else:
            label = label[train_len:]
        a_mask = np.zeros(len(label), dtype=bool)
        for i in range(len(label)):
            if int(label[i, 1]) in range(age_min, age_max):
                a_mask[i] = True
        self.label = label[a_mask]
        self.length = len(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_name = os.path.join(self.image_dir, self.label[index][0])
        if self.training_set and self.obscure_age:
            age_val = int(self.label[index][1]) + np.random.randint(-1, 1)
        else:
            age_val = int(self.label[index][1])
        age = torch.tensor(age_val, dtype=torch.long)

        image = Image.open(img_name).convert('RGB')
        img = self.transform(image)

        return img, age
