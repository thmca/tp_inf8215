import pandas as pd
from sklearn import tree


def train(SK_model, x_train, y_train):
    model = SK_model
    model = model.fit(x_train, y_train)

    return model


def predict(model, data_to_predict):
    predictions = model.predict(data_to_predict)
    predictions_df = pd.DataFrame(predictions)

    return predictions_df


def validate_predictions(SK_model, x_train, y_train, x_validate):
    model = train(SK_model, x_train, y_train)
    predictions = predict(model, x_validate)
    return predictions