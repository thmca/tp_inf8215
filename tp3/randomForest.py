
from sklearn.ensemble import RandomForestClassifier

from ScikitModel import predict


def validate_predictions(x_train, y_train, x_validate, y_validate):

    rndTree = RandomForestClassifier()

    rndTree.fit(x_train,y_train)

    RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                           max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                           bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=10,
                           warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    score = rndTree.score(x_validate, y_validate)

    predictions = predict(rndTree, x_validate)

    print(f'Random Forest Tree - Accuracy: {score}')

    return predictions, rndTree

def submission_predictions(x_all,y_all, test_df):

    rndTree = RandomForestClassifier()

    rndTree.fit(x_all, y_all)

    RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                           max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                           bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=10,
                           warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)


    predictions = predict(rndTree, test_df)


    return predictions, rndTree