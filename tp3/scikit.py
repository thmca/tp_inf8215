# pip install -U scikit-learn
# pip3 install numpy
# pip3 install scipy

# import sklearn as skl
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from sklearn import tree, neural_network, naive_bayes, mixture, linear_model, decomposition
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA


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

def train(SK_model, x_train, y_train):

    model = SK_model
    model = model.fit(x_train, y_train)

    return model

def predict(model,data_to_predict):
    predictions = model.predict(data_to_predict)
    predictions_df = pd.DataFrame(predictions)

    return predictions_df


def submit(predictions_dataframe):
    predictions_dataframe.columns = ["status"]
    predictions_dataframe.index.name = "idx"
    map = {True: 'phishing', False: 'legitimate'}
    predictions_df = predictions_dataframe.replace(map)
    predictions_df.to_csv(submission_file)

def testModel(SK_model, name) :
    model = train(SK_model, x_train, y_train)
    predictions = predict(model, x_validate)
    print(name, " : ", f1_score(y_validate, predictions))

    return model, predictions


def normal_scaling(train_set, validation_set, test_set):
    scaler = StandardScaler()
    scaler.fit(train_set)
    new_train = scaler.transform(train_set)
    new_validate = scaler.transform(validation_set)
    new_test = scaler.transform(test_set)

    return new_train, new_validate, new_test


def scaling_test_model(SK_model, name):
    x_train_scaled, x_validate_scaled, test_set = normal_scaling(x_train, x_validate, test_df)
    model = train(SK_model, x_train_scaled, y_train)
    predictions = predict(model, x_validate_scaled)
    print("normal scaling + ", name, " : ", f1_score(y_validate, predictions))

    return model, predictions, test_set

def apply_PCA(train_set, validation_set, test_set) :
    train_scaled, validate_scaled, test_scaled = normal_scaling(train_set, validation_set, test_set)
    pca = PCA(n_components='mle')
    pca.fit(train_scaled)
    new_train = pca.transform(train_scaled)
    new_validate = pca.transform(validate_scaled)
    new_test = pca.transform(test_scaled)

    return new_train, new_validate, new_test


def PCA_test_model(SK_model, name) :
    x_train_scaled, x_validate_scaled, test_set = apply_PCA(x_train, x_validate, test_df)
    model = train(SK_model, x_train_scaled, y_train)
    predictions = predict(model, x_validate_scaled)
    print("PCA + ", name, " : ", f1_score(y_validate, predictions))

    return model, predictions, test_set

# testModel(neural_network.MLPClassifier(random_state=1), "MLPClassifier neural network")
model, validate_predictions, normalized_test = scaling_test_model(neural_network.MLPClassifier(random_state=1), "MLPClassifier neural network")
model, validate_predictions, normalized_test = PCA_test_model(neural_network.MLPClassifier(random_state=1), "MLPClassifier neural network")

test_predictions = predict(model, normalized_test)
submit(test_predictions)

# testModel(tree.DecisionTreeClassifier(), "decision Tree classifier")
# testModel(tree.DecisionTreeRegressor(), "decision Tree classifier")
# scaling_test_Model(tree.DecisionTreeClassifier(), "decision Tree classifier")

# testModel(neural_network.MLPClassifier(hidden_layer_sizes=(3, 1), random_state=1), "MLPClassifier neural network")

# testModel(naive_bayes.BernoulliNB(), "Bernouilli naive bayes")
# testModel(naive_bayes.GaussianNB(), "Gaussian naive bayes")
# testModel(naive_bayes.CategoricalNB(), "Categorical naive bayes")
# testModel(naive_bayes.ComplementNB(), "ComplementNB naive bayes")
# testModel(naive_bayes.MultinomialNB(), "Multinomial naive bayes")

# testModel(linear_model.LogisticRegression(), "Logistic regression")





