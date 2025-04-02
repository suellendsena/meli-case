import pandas as pd
import numpy as np
import json

from sklearn.base import BaseEstimator, TransformerMixin


def get_quarter(month: str) -> str:
    """Build quarter feature by month of the year"""
    if month in ['jan', 'feb', 'mar']:
        return '1Q'
    elif month in ['apr', 'may', 'jun']:
        return '2Q'
    elif month in ['jul', 'aug', 'sep']:
        return '3Q'
    elif month in ['oct', 'nov', 'dec']:
        return '4Q'
    else:
        return 'unknown'


def calc_contacts_tendency(num: float, denom: float) -> float:
    """Build ratio feature, avoiding math numerical errors"""
    if np.isnan(denom):
        return np.nan
    elif denom == 0:
        return 0
    else:
        return num / denom


def categorize_employment(job: str) -> str:
    not_employed = ['retired', 'student', 'unemployed']

    """Classify employment status"""
    if job in not_employed:
        return 'not_employed'
    elif job == 'unknown':
        return 'unknown'
    else:
        return 'employed'


class BuildFeatures(BaseEstimator, TransformerMixin):
    """
    Class to build features in production or training pipelines.
    """

    def __init__(self, training=False):
        super().__init__()
        self.training = training

    def __repr__(self):
        return "Object intended to build features"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.build_features(X)
        return X

    def build_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["quarter"] = X["month"].apply(get_quarter)
        X["contacts_tendency"] = X.apply(lambda x: calc_contacts_tendency(x["campaign"], x["previous"]), axis=1)
        X["employment_status"] = X["job"].apply(categorize_employment)
        X["was_contacted_before"] = X["pdays"] != 999

        if self.training:
            return X
        else:
            return X


class Json2DF:  
    def __init__(self):
        """Class that parses the json (payload) to a dataframe type"""
    
    def fit(self, X, y):
        pass

    def __repr__(self):
        return "Json parser"
    
    def transform(self, input_data):
        if type(input_data) == str:
            json_input = str(input_data)
            return pd.json_normalize(json.loads(json_input)).replace(to_replace=[None], value=np.nan)
        elif type(input_data) == pd.DataFrame:
            return input_data


class Selector:
    def __init__(self, features: list, target: str, mode='train'):
        self.features = features
        self.target = target
        self.mode = mode
    
    def __repr__(self):
        return f"Feature Selector. Mode: {self.mode}. Features: {self.features}"
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        if self.mode == 'train':
            return X[self.features] 
        elif self.mode == 'inference':
            return X[self.features] 

    
class FillStringMissing:
    def __init__(self, cols_to_adjust):
        """Fill missing strings as '<unknown>'"""
        self.cols_to_adjust = cols_to_adjust

    def __repr__(self):
        return f"Fill missing strings as '<unknown>'. Columns: {self.cols_to_adjust}"
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].fillna('<unknown>')
        return X



class NormalizeLowerString:
    def __init__(self, cols_to_adjust):
        self.cols_to_adjust = cols_to_adjust
    
    def normalize(self, x):
        y = x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        return y
    
    def __repr__(self):
        return f"Convert words to lower case and remove accents. Columns: {self.cols_to_adjust}"
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].apply(self.normalize)
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].apply(lambda x: x.str.lower())
        return X



class BoolHandler:
    def __init__(self, cols_to_adjust):
        self.cols_to_adjust=cols_to_adjust

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].replace({
            'yes': True,
            'no': False,
            'unknown': None
        })
        
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].astype(float).fillna(-1.0)
        return X
    


class FillNull:
    def __init__(self, cols_to_adjust):
        """Fill missing given numbers as -999"""
        self.cols_to_adjust=cols_to_adjust
    
    def __repr__(self):
        return f"Fill missing given numbers as. Columns {self.cols_to_adjust}"
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].fillna(-999)
        return X


class ConverteFloat:
    def __init__(self):
        pass

    def __repr__(self):
        return "Float converter"
    
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.astype(float)
        return X
