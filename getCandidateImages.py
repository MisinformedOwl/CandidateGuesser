import urllib.request
import os
import pandas as pd
from time import sleep
import random

def getImage(link: str, id: str):
    urllib.request.urlretrieve(link, f"Candidates/{id}.jpg")


def checkFolderExists():
    path = os.getcwd()+"\\Candidates"
    print(f"Checking for: {path}")
    if not os.path.isdir(path):
        os.mkdir(path)


def loadData() -> pd.DataFrame:
    return pd.read_csv("candidatesData.csv")


def collectImages(data: pd.DataFrame):
    wantedParties = set({"Conservative and Unionist Party", "Reform UK", "Green Party", "Labour Party", "Liberal Democrats"})
    newData = pd.DataFrame(columns=["id", "name", "party"])
    try:
        for _, candidate in data.iterrows():
            if pd.isna(candidate["image"]):
                print(f"No image for candidate {candidate["person_id"]}")
                continue
            if not candidate["party_name"] in wantedParties:
                print(f"Unwanted party for candidate {candidate["person_id"]}")
                continue
            newData.loc[len(newData)] = [candidate["person_id"], candidate["person_name"], candidate["party_name"]]
            print(candidate["image"])
            getImage(candidate["image"], candidate["person_id"])
            sleep(1+random.random()/2)
    except KeyboardInterrupt:
        print("Ending search.")
    except Exception as e:
        print(f"Unknown exception {e.__class__.__name__}")
    newData.to_csv("candidates.csv")


if __name__ == "__main__":
    checkFolderExists()
    data = loadData()
    collectImages(data)