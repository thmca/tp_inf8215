# pip install -U scikit-learn
# pip3 install numpy
# pip3 install scipy

# import sklearn as skl
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from sklearn import tree


data_file = open("data/train.csv", "r")
test_file = open("data/test.csv", "r")
submission_file = open("data/submission.csv", "w")

data_df = pd.read_csv(data_file)
data_df["status"] = (data_df["status"] == "phishing").astype(bool)
test_df = pd.read_csv(test_file)
test_df = test_df.iloc[:, 1:]

train_df, validate_df = train_test_split(data_df, test_size=0.2)

y_train = train_df.iloc[:, (train_df.shape[1]-1)]
x_train = train_df.iloc[:, 1:(train_df.shape[1]-1)]
y_validate = validate_df.iloc[:, (train_df.shape[1]-1)]
x_validate = validate_df.iloc[:, 1:(train_df.shape[1]-1)]


decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(x_train, y_train)

# tree_prediction_validate = decision_tree.predict(x_validate)
tree_prediction_validate = decision_tree.predict(test_df)

tree_prediction_df = pd.DataFrame(tree_prediction_validate)
tree_prediction_df.columns = ["status"]
tree_prediction_df.index.name = "idx"
map = {True: 'phishing', False: 'legitimate'}
tree_prediction_df = tree_prediction_df.replace(map)
tree_prediction_df.to_csv(submission_file)

# print("f1 score :")
# print(f1_score(y_validate, tree_prediction_validate))

