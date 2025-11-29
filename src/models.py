from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
import pandas as pd

def get_models():
    """
    Returns a dictionary of initialized models.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }
    return models

def get_param_grids():
    """
    Returns a dictionary of parameter grids for hyperparameter tuning.
    """
    grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5]
        },
        'CatBoost': {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1],
            'depth': [4, 6]
        }
    }
    return grids

def tune_model(model, param_grid, X, y):
    """
    Performs hyperparameter tuning using GridSearchCV.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_models_cv(models, X, y):
    """
    Evaluates models using 5-fold Stratified Cross-Validation.
    
    Args:
        models (dict): Dictionary of models.
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        
    Returns:
        pd.DataFrame: Summary of performance metrics.
    """
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
            results[name] = {
                'Accuracy': scores['test_accuracy'].mean(),
                'Precision': scores['test_precision'].mean(),
                'Recall': scores['test_recall'].mean(),
                'F1 Score': scores['test_f1'].mean()
            }
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {
                'Accuracy': None, 'Precision': None, 'Recall': None, 'F1 Score': None
            }
            
    return pd.DataFrame(results).T
