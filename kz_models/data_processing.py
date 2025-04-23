import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io, transform as sk_transform

class ChestScanDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_col="Lung Opacity", transform=None):
        self.pd_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.pd_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.pd_df.iloc[index]

        # load image
        img_path = os.path.join(self.root_dir, row["Path"])
        image = io.imread(img_path)

        # label (custom column)
        label = row[self.label_col]
        label = 0.0 if pd.isna(label) else float(label)

        # age
        age = row["Age"]
        age = 0.0 if pd.isna(age) else float(age)

        # aex (Male=0, Female=1, Unknown=-1)
        sex = row["Sex"]
        sex = 0.0 if sex == "Male" else 1.0 if sex == "Female" else -1.0

        sample = {"image": image, "label": label, "age": age, "sex": sex}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, age, sex = sample["image"], sample["label"], sample["age"], sample["sex"]
        h, w = image.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        image = sk_transform.resize(image, (int(new_h), int(new_w)))
        return {"image": image, "label": label, "age": age, "sex": sex}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def __call__(self, sample):
        image, label, age, sex = sample["image"], sample["label"], sample["age"], sample["sex"]
        h, w = image.shape
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top:top + new_h, left:left + new_w]
        return {"image": image, "label": label, "age": age, "sex": sex}

class ToTensor(object):
    def __call__(self, sample):
        image, label, age, sex = sample["image"], sample["label"], sample["age"], sample["sex"]
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]  # add channel dimension
        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.tensor(label).float(),
            "age": torch.tensor(age).float(),
            "sex": torch.tensor(sex).float()
        }
