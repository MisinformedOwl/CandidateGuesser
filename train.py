import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class dataset(Dataset):
    def __init__(self, paths: list[str], labels, transform=None):
        self.imagePaths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.imagePaths[index]
        image = Image.open(path).convert("RGB")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)

dir = "./Candidates"

candidatePaths = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".jpg")]

labels = grabLabels()