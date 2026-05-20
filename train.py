import torch
from torch import Generator
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

#------------------------------------Globals----------------------------------------------

partyDict = {"Conservative and Unionist Party" : 0, "Green Party": 1, "Labour Party": 2, "Liberal Democrats": 3, "Reform UK" : 4}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)

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
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),   # Makes it robust to input size
            nn.Flatten(),
            nn.Linear(256*4*4, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
#---------------------------------Functions--------------------------------------

def loadDataset(transforms: v2.Compose = None) -> candidateDataset:
    assert transforms != None
    print("Loading csv.")
    data = pd.read_csv("candidates.csv", index_col=0)
    return candidateDataset(data["imageLoc"],data["party"],transform=transforms), setClassWeights(data)


def setClassWeights(data: pd.DataFrame):
    counts = data.groupby("party").size()
    weights = counts.max()/counts
    weights = weights/weights.max()
    return torch.Tensor(weights).to(device)


#----------------------------Main------------------------------------------

if __name__ == "__main__":
    print(f"Using {device} for training.")

    transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop((256,256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    dataset, classWeights = loadDataset(transforms)
    trainData, testData = random_split(dataset, [0.8,0.2], generator=Generator().manual_seed(42))
    del dataset

    trainLoader = DataLoader(
        dataset=trainData,
        batch_size=32,
        num_workers=2,
        shuffle=True,
        persistent_workers=True
    )

    model = ModelBuild().to(device)

    print(f"Class weights are... {classWeights}")
    lossfn = nn.CrossEntropyLoss(weight=classWeights)
    epochs = 20
    lr= 0.0001
    optim = SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=1e-4)
    runningLoss = []
    

    print("Training...")
    model.train()
    for e in tqdm(range(epochs),desc="Training epochs"):
        epochloss = 0
        for batch in trainLoader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            
            outputs = model(images)
            loss = lossfn(outputs, labels)
            epochloss += loss
            loss.backward()
            optim.step()
        runningLoss.append(epochloss)
    
    del trainLoader
    testLoader = DataLoader(
        dataset=testData,
        batch_size=32,
        num_workers=2,
        shuffle = True
    )

    print("Testing...")
    model.eval()
    trueCount = 0
    totalCount = 0
    for batch in testLoader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        outputs = torch.argmax(outputs, dim=1)
        print(outputs)
        for output, label in zip(outputs, labels):
            totalCount+=1
            if torch.equal(output,label):
                trueCount += 1
    del testLoader
    print(f"trueCount: {trueCount}")
    print(f"totalCount: {totalCount}")
    print(f"Accuracy: {trueCount/totalCount}")