import urllib.request

def getImage(link: str, id: str):
    urllib.request.urlretrieve(link, f"Candidates/{id}.jpg")

if __name__ == "__main__":
    getImage("https://candidates.democracyclub.org.uk/media/images/people/82796/b91d8409-bc0a-4520-93ba-813f34ff1118.png", 0)