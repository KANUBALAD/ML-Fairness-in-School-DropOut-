from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def logistic_regression(X, y):
    lrmodel = LogisticRegression()
    lrmodel.fit(X, y)
    return lrmodel

def decision_tree(X, y):
    decision_model = DecisionTreeClassifier()
    decision_model.fit(X, y)
    return decision_model

def random_forest(X, y):
    rfmodel = RandomForestClassifier()
    rfmodel.fit(X, y)
    return rfmodel


def run_model(config, split_data):
    X_train, X_test, y_train, y_test = split_data['X_train'], split_data['X_test'], split_data['y_train'], split_data['y_test']
    
    if config['model'] == 'logistic_regression':
        model_lr = logistic_regression(X_train, y_train)
        score = model_lr.score(X_test, y_test)
        print(f"{config['model']} accuracy: {score}")
        return {config['model']: score}
    
    elif config['model'] == 'decision_tree':
        model_dt = decision_tree(X_train, y_train)
        score = model_dt.score(X_test, y_test)
        print(f"{config['model']} accuracy: {score}")
        return {config['model']: score}
    
    elif config['model'] == 'random_forest':
        model_rf = random_forest(X_train, y_train)
        score = model_rf.score(X_test, y_test)
        print(f"{config['model']} accuracy: {score}")
        return {config['model']: score}
    
    elif config['model'] == 'compare':
        models_all = {
            'logistic_regression': logistic_regression(X_train, y_train),
            'decision_tree': decision_tree(X_train, y_train),
            'random_forest': random_forest(X_train, y_train)
        }
        scores = {}
        for name, model_item in models_all.items():
            score = model_item.score(X_test, y_test)
            print(f"{name} accuracy: {score}")
            scores[name] = score
        return scores
    