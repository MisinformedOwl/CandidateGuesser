import torch
from torch import Generator
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import torchvision
from torchvision.transforms import v2

#------------------------------------Globals----------------------------------------------

partyDict = {"Conservative and Unionist Party" : 1, "Reform UK" : 2, "Green Party": 3, "Labour Party": 4, "Liberal Democrats": 5}

#-----------------------------------Classes-----------------------------------------------

class candidateDataset(Dataset):
    """
    Creates a custom dataset compatible with torch's DataLoader to avoid putting too much pressure on my ram when training potentially thousands of images.
    """
    def __init__(self, paths: list[str], labels, transform=None):
        self.imagePaths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        """
        Grabs the image path and opens the image to convert it to a image and then a tensor.
        """
        path = self.imagePaths[index]
        #Read image, set to RGB to avoid alpha channel. Then convert to a float so it can be normalised for use in a tensor.
        image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB).float() / 255.0
        resize = v2.Resize((224,224), antialias=True)
        image = resize(image)
        if self.transform:
            image = self.transform(image)

        label = partyDict[self.labels[index]]

        
        return image, torch.tensor(label)


class ModelBuild(nn.Module):
    """
    Responcible for building the neural network.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2,3,5)
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        """
        This is where the data is put forward.
        """
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        return x

#---------------------------------Functions--------------------------------------

def loadDataset() -> candidateDataset:
    print("Loading csv.")
    data = pd.read_csv("candidates.csv", index_col=0)
    return candidateDataset(data["imageLoc"],data["party"])

#----------------------------Main------------------------------------------

if __name__ == "__main__":
    dataset = loadDataset()
    trainData, testData = random_split(dataset, [0.8,0.2], generator=Generator().manual_seed(42))

    trainLoader = DataLoader(
        dataset=trainData,
        batch_size=32,
        num_workers=2,
        shuffle = True
    )

    model = ModelBuild()

    lossfn = nn.CrossEntropyLoss()
    epochs = 10
    lr= 0.001
    optim = SGD(model.parameters(),lr=lr, momentum=0.9)

    model.train()
    for e in range(epochs):
        for batch in trainLoader:
            print(type(batch), batch)
            os.exit(0)
            images, labels = batch
            optim.zero_grad()
            # get results
            outputs = model(batch)
            loss = lossfn(outputs, labels)
            loss.backward()
            optim.step()
            #loss data analysis.