import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

#------------------------------------Globals----------------------------------------------

partyDict = {"Conservative and Unionist Party" : 1, "Reform UK" : 2, "Green Party": 3, "Labour Party": 4, "Liberal Democrats": 5}

#-----------------------------------Classes-----------------------------------------------

class candidateDataset(Dataset):
    def __init__(self, paths: list[str], labels, transform=None):
        self.imagePaths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        path = self.imagePaths[index]
        image = Image.open(path).convert("RGB")
        label = partyDict[self.labels[index]]

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)

#---------------------------------Functions--------------------------------------

def testgen():
    for index in range(len(dataset)):
        print(f"Index now at {index}")
        data = dataset.__getitem__(index)
        print(data)
        yield data[0], data[1]

#----------------------------Main------------------------------------------

dir = "./Candidates"

print("Loading csv.")
data = pd.read_csv("candidates.csv", index_col=0)

dataset = candidateDataset(data["imageLoc"],data["party"])