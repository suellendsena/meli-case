import pandas as pd
import numpy as np

import xgboost as xgb
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

def objective_lgbm(trial, X_train, y_train):
    """
    Função objetivo a ser otimizada:
    1) Cria um grid de parâmetros
    2) Estancia o modelo a ser utilizado (LGBM)
    3) Por meio da validação cruzada (estratificada, dado que a target é desbalanceada), retorna a média da métrica (AUC) dos folds de validação.
    """

    auc_scores = []
    params_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'random_state': 98,
            'verbosity': -1
        }
    
    model = LGBMClassifier(**params_grid)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=98)

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
        auc_scores.append(roc_auc)

    return np.mean(auc_scores)




def objective_xgboost(trial, X_train, y_train):
    
    """
    Função objetivo a ser otimizada:
    1) Cria um grid de parâmetros
    2) Estancia o modelo a ser utilizado (XGBoost)
    3) Por meio da validação cruzada (estratificada, dado que a target é desbalanceada), retorna a média da métrica (AUC) dos folds de validação.
    """

    auc_scores = []
    params_grid = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e2),
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 1e2),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 1e2),
        'random_state': 98,
        'verbosity': 0,
        'objective': 'binary:logistic' 
    }

    model = xgb.XGBClassifier(**params_grid)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=98)

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
        auc_scores.append(roc_auc)

    return np.mean(auc_scores)



def objective_xgboost_reg(trial, X_train, y_train):
    """
    Função objetivo para otimização de hiperparâmetros do XGBoost no contexto de regressão:
    1) Cria um grid de parâmetros.
    2) Instancia o modelo XGBoost para regressão.
    3) Utiliza validação cruzada (K-Fold) e retorna a média da métrica (MSE) dos folds de validação.
    """

    mse_scores = []
    params_grid = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e2),
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 1e2),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'random_state': 98,
        'verbosity': 0,
        'objective': 'reg:squarederror'
    }

    model = xgb.XGBRegressor(**params_grid)

    kf = KFold(n_splits=5, shuffle=True, random_state=98)

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_val_fold)

        mse = mean_squared_error(y_val_fold, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores)