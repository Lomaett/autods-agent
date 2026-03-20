import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def preprocess(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target])
        X = pd.get_dummies(X, drop_first=True)  # Handle categorical variables

        if self.df[self.target].dtype == 'object':
            y = LabelEncoder().fit_transform(self.df[self.target])  # Encode target if it's categorical
        else:
            y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test
    
    def classify(self, X_train, X_test, y_train, y_test):

        models = {
            "random_forest": RandomForestClassifier(),
            "logistic_regression": LogisticRegression(),
        }
        
        try:
            from xgboost import XGBClassifier
            xgboost_available = True
        except ImportError:
            xgboost_available = False

        if xgboost_available:
            models["xgboost"] = XGBClassifier()

        results = {}

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds)
            class_report = classification_report(y_test, preds)

            results[name] = {"score": score, "classification_report": class_report}

        return results
    
    def regress(self, X_train, X_test, y_train, y_test):

        models = {
            "random_forest": RandomForestRegressor(),
            "linear_regression": LinearRegression(),
        }

        try:
            from xgboost import XGBRegressor
            xgboost_available = True
        except ImportError:
            xgboost_available = False
        
        if xgboost_available:
            models["xgboost"] = XGBRegressor()

        results = {}

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            results[name] = {"mse": mse, "r2": r2}

        return results
    
    def train(self):
        X_train, X_test, y_train, y_test = self.preprocess()

        if self.df[self.target].dtype == 'object':
            return self.classify(X_train, X_test, y_train, y_test)
        else:
            return self.regress(X_train, X_test, y_train, y_test)