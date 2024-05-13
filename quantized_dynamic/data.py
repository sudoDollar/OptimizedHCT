import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode

import pandas as pd
from PIL import Image
import PIL.Image


class LIU4K(Dataset):

    def __init__(self, annotations_file, train:bool):
        self.img_labels = pd.read_csv(annotations_file, header=None, delimiter=' ')
        self.train = train
        # PIL.Image.MAX_IMAGE_PIXELS = None
        self.transform1 = v2.Compose([
                            v2.ToImage(),
                            v2.CenterCrop((8000, 8000)),
                            v2.Resize((4000,4000), InterpolationMode.BICUBIC),
                            v2.ToDtype(torch.float32, scale=True),
                            # v2.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
        self.transform2 = v2.Compose([
                            v2.ToImage(),
                            v2.CenterCrop((8000, 8000)),
                            v2.Resize((4000,4000), InterpolationMode.BICUBIC),
                            v2.RandomHorizontalFlip(0.5),
                            v2.ToDtype(torch.float32, scale=True),
                            # v2.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
        
        self.transform3 = v2.Compose([
                            v2.ToImage(),
                            v2.CenterCrop((4000, 4000)),
                            v2.RandomHorizontalFlip(0.5),
                            v2.ToDtype(torch.float32, scale=True),
                            # v2.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
        
        self.transform4 = v2.Compose([
                            v2.ToImage(),
                            v2.CenterCrop((4000, 4000)),
                            v2.ToDtype(torch.float32, scale=True),
                            # v2.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        # image = read_image(img_path)
        img_path = "../" + img_path
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        w, h = image.size
        if self.train:
            if w <= 4500 and h <= 4500:
                image = self.transform3(image)
            else:
                image = self.transform2(image)
        else:
            if w <= 4500 and h <= 4500:
                image = self.transform4(image)
            else:
                image = self.transform1(image)
        return image, label