
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

from scikit import predict


def validate_predictions(x_train, y_train, x_validate, y_validate):

    GNB = GaussianNB()

    GNB.fit(x_train,y_train)

    score = GNB.score(x_validate, y_validate)

    predictions = predict(GNB, x_validate)
    print(f'Gaussian Naive Bayes - Accuracy: {score}')
    return predictions


def submission_predictions(x_all, y_all, test_df):
    GNB = GaussianNB()

    GNB.fit(x_all, y_all)

    predictions = predict(GNB, test_df)

    return predictions