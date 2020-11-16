import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


data_file = open("data/train.csv", "r")
test_file = open("data/test.csv", "r")
submission_file = open("data/submission.csv", "w")

df=pd.read_csv(data_file)

df.head()

#True = phishing
df['phishing']=pd.get_dummies(df['status'], drop_first=True)

df=df.drop('status', axis=1)

df.info()
print(df)

