# The Candidate Detector

### Abandoned
After finally making the AI, i realise that there is no real way for the AI to discern the data, similarly to how me and my friends were unable to ourselves. And so therefore, given the quality of images, aswell as the lack of other inforamtion to use inside of the training. I have decided to stop working on this and return to learning about data engineering.

### Why did i make this?
Me and some friends were on a website where we had to guess based off a picture, which political party a candidate belonged to. And throughout this i could not stop thinking about if i could make convolutional neural network detect this, and then further more. Find out what party it thinks me and my friends are part off.

### Where did i get the data?
I am using a csv file acquired from https://candidates.democracyclub.org.uk/data/?election_date=2026-05-07&election_id=%5Elocal.*&format=csv&field_group=results. This csv file provides more than enough information for me. I have the candidate party membership, their id for marking images with, and a link to their image.

## Running this on your local machine.
To run this on your local machine, you will have to open a terminal to the folder you put the code in. And type
```
pip install -r requirements.txt
```
to install the requirements of the program. (Not in yet.)

Then, you will want to firstly run
```
python getCandidateImages.py
```
This will using the data inside of the candidatesData.csv pull the images from the website so they can be used later in train.py. This code will take a long time, as to avoid getting IP banned from collecting. When you feel you have collected enough, simply hit ctrl+c to keyboard interrupt, the program will save what you have collected and exit.

Then, once you have collected a sufficient amount. simpy run the following command, and watch it train. And showcase the results.
```
python train.py
```
