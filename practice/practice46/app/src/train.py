# app/src/train.py

from sklearn.model_selection import train_test_split
from . import data_preprocessing
from . import model

def train_model(df):
    """
    Train the model.
    """
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = model.create_model()
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    print(f"Model accuracy: {score}")
    
    return clf, score
