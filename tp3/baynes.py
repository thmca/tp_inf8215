
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

def validate_predictions(x_train, y_train, x_validate, y_validate):

    GNB = GaussianNB()

    GNB.fit(x_train,y_train)
    train_score = GNB.score(x_train,y_train)
    test_score = GNB.score(x_validate,y_validate)
    print(f'Gaussian Naive Bayes : Training score - {train_score} - Test score - {test_score}')