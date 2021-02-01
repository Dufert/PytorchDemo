import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

class mDataset(data.Dataset):

    def __init__(self, Path):
        self.imgClass_Data_label = []
        imgClassAllPath = [Path + x for x in os.listdir(Path) if os.path.isdir(Path + x)]
        for classNum, imgClassPath in enumerate(imgClassAllPath):
            for imgName in os.listdir(imgClassPath):
                self.imgClass_Data_label.append([imgClassPath+"/"+imgName, classNum])

    def __len__(self):
        return len(self.imgClass_Data_label)

    def __getitem__(self, idx):
        [imgAbsPath, label] = self.imgClass_Data_label[idx]
        img = Image.open(imgAbsPath)
        transformsImage = transforms.Compose(
                                     [transforms.Scale([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
        img = transformsImage(img)

        return img, label