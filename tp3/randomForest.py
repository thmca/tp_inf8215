
from sklearn.ensemble import RandomForestClassifier

from ScikitModel import predict


def validate_predictions(x_train, y_train, x_validate, y_validate):

    rndTree = RandomForestClassifier()

    rndTree.fit(x_train,y_train)

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                           oob_score=False, random_state=None, verbose=0,
                           warm_start=False)

    test_score = rndTree.score(x_validate, y_validate)

    predictions = predict(rndTree, x_validate)

    print(f'- Test score - {test_score}')

    return predictions
