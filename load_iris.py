def load_data():
    import pandas as pd
    from sklearn.datasets import load_iris

    data, label = load_iris(return_X_y=True, as_frame=True)
    return data, label

def prepare_data(data, label):
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(data.values, label.values.reshape(-1,1), test_size=0.2, random_state=2023)

    return train_X, test_X, train_y, test_y


def model_pipeline(train_X, test_X, train_y, test_y):
    import joblib
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    model_pipeline = Pipeline([("clf",DecisionTreeClassifier(random_state=2023))])
    model_pipeline.fit(train_X, train_y)
    pred = model_pipeline.predict(test_X)
    print("Test Accuracy :", np.sum(pred.reshape(-1,1)==test_y)/test_y.shape[0]*100)
    joblib.dump(model_pipeline, './model_pipelie.pkl')