import urllib.request
import os
import pandas as pd
from time import sleep
import random

#---------------------------------------------------------------------------------------------

def getImage(link: str, id: str) -> str:
    path = f"Candidates/{id}.jpg"
    urllib.request.urlretrieve(link, path)
    return path


def checkFolderExists():
    path = os.getcwd()+"\\Candidates"
    print(f"Checking for: {path}")
    if not os.path.isdir(path):
        os.mkdir(path)


def collectImages(data: pd.DataFrame):
    wantedParties = set({"Conservative and Unionist Party", "Reform UK", "Green Party", "Labour Party", "Liberal Democrats"})
    if os.path.exists("candidates.csv"):
        print("Loading existing data for candidates.")
        newData = pd.read_csv("candidates.csv", index_col=0)
    else:
        print("Creating empty data for candidates.")
        newData = pd.DataFrame(columns=["id", "name", "party", "imageLoc"])
    
    caughtUp = True
    targetID = -1
    if newData.shape[0] > 0:
        caughtUp = False
        targetID = newData.loc[len(newData)-1]["id"]
    
    try:
        for _, candidate in data.iterrows():
            if not caughtUp:
                if candidate["person_id"] != targetID:
                    print("Continue to next candidate")
                    continue
                else:
                    caughtUp = True
                    continue
            if pd.isna(candidate["image"]): # Check if candidate has an image.
                print(f"No image for candidate {candidate["person_id"]}")
                continue
            if not candidate["party_name"] in wantedParties: # Check if they are part of the 5 parties i'm checking for.
                print(f"Unwanted party for candidate {candidate["person_id"]}")
                continue
            path = getImage(candidate["image"], candidate["person_id"])
            newData.loc[len(newData)] = [candidate["person_id"], candidate["person_name"], candidate["party_name"], path]
            print(candidate["image"])
            sleep(1+random.random()/2)
    except KeyboardInterrupt:
        print("Ending search.")
    except Exception as e:
        print(f"Unknown exception {e.__class__.__name__}")
        print(e)
    newData.to_csv("candidates.csv")

#---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    checkFolderExists()
    data = pd.read_csv("candidatesData.csv")
    collectImages(data)