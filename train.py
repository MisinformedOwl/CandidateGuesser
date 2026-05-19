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

partyDict = {"Conservative and Unionist Party" : 0, "Reform UK" : 1, "Green Party": 2, "Labour Party": 3, "Liberal Democrats": 4}
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
        image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB).float().to(device) / 255.0
        if self.transform:
            image = self.transform(image)

        label = partyDict[self.labels[index]]

        
        return image, torch.tensor(label).to(device)


class ModelBuild(nn.Module):
    """
    Responcible for building the neural network.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3)
        self.conv3 = nn.Conv2d(32,32,3)

        self.lin1 = nn.Linear(30*30*32, 256)
        self.lin2 = nn.Linear(256, 5)
    
    def work(self, x):
        """
        This is where the data is sent, not in the form of a list.
        """
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.lin1(x))
        x = torch.dropout(x,0.2,self.training)
        x = self.lin2(x)
        return x

    def forward(self, x):
        """
        This is where the data is put forward.
        It is put into a for each loop to extract the data.
        """
        outputs = torch.Tensor().to(device)
        outputs = self.work(x)
        return outputs

#---------------------------------Functions--------------------------------------

def loadDataset(transforms: v2.Compose = None) -> candidateDataset:
    assert transforms != None
    print("Loading csv.")
    data = pd.read_csv("candidates.csv", index_col=0)
    return candidateDataset(data["imageLoc"],data["party"],transform=transforms)

#----------------------------Main------------------------------------------

if __name__ == "__main__":

    transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop((256,256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    dataset = loadDataset(transforms)
    trainData, testData = random_split(dataset, [0.8,0.2], generator=Generator().manual_seed(42))
    del dataset

    trainLoader = DataLoader(
        dataset=trainData,
        batch_size=32,
        num_workers=2,
        shuffle = True,
        persistent_workers=True
    )

    model = ModelBuild().to(device)

    lossfn = nn.CrossEntropyLoss()
    epochs = 10
    lr= 0.001
    optim = SGD(model.parameters(),lr=lr, momentum=0.9)
    runningLoss = []

    print("Training...")
    model.train()
    for e in tqdm(range(epochs),desc="Training epochs"):
        epochloss = 0
        for batch in trainLoader:
            images, labels = batch
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
        outputs = model(batch[0])
        outputs = torch.argmax(outputs, dim=1)
        for output, label in zip(outputs, batch[1]):
            totalCount+=1
            if torch.equal(output,label):
                trueCount += 1
    print(f"trueCount: {trueCount}")
    print(f"totalCount: {totalCount}")
    print(f"Accuracy: {trueCount/totalCount}")