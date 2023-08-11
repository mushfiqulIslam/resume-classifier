# resume-classifier

In the example.env file we have all our configuration related
variables. To  train and then predict we need to set  
```
TRAIN_FIRST=ture # false means will use existing binary file
```
Provide csv file links for training on
```
TRAINING_CSV=M:/resume-classifier/Resume/Resume.csv
```

To get output from the classifier we must need to set the following value
```
DATA_PATH_TO_CLASSIFY=M:/resume-classifier/data/ACCOUNTANT
```
Before running scripts please set up the environment. Commands are given below
```
python3 -m venv venv
source venv/bin/activate
pip install -r reuirements.txt
```

Finally, run this with this command
```
python3 script.py
```