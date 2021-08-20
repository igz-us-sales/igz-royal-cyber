import sys
import pickle
import pandas as pd
from typing import Tuple

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_data(test_size:float=0.3):
    """
    Load test iris dataset. Split into X_train, X_test, y_train, y_test.
    
    :param test_size: Percentage of dataset to use for test set.
    
    :returns:         X_train, X_test, y_train, y_test
    """
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def str_to_class(str:str):
    """
    Turns a string into Python class. Used to dynamically load Sklearn model classes.
    Note: Desired class must be imported at top of script.
    
    :param str: String to dynamically load as Python class.
    
    :returns:   Python class corresponding to string.
    """
    return eval(str)

def build_model(model_class:str, model_params:dict):
    """
    Build Sklearn model of certain type with parameters.
    
    :param model_class:  Sklearn class to create.
    :param model_params: Dict of model parameters to use.
    
    :returns:            Newly built sklearn model.
    """
    model_class = str_to_class(model_class)
    return model_class(**model_params)

def train_model(
    model,
    hyperparameters:dict,
    X_train: pd.DataFrame,
    y_train: pd.Series
):
    """
    Train Sklearn model with random hyperparameter search.
    
    :param model:           Sklearn model to train.
    :param hyperparameters: Hyperparameter grid to search.
    :param X_train:         Training data.
    :param y_train:         Training labels.
    
    :returns:               Best trained Sklearn model.
    """
    clf = RandomizedSearchCV(model, hyperparameters, random_state=0)
    search = clf.fit(X_train, y_train)
    return search.best_estimator_

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates trained SKlearn model with common metrics.
    
    :param model:  Trained Sklearn model.
    :param X_test: Test data to evaluate with.
    
    :returns:      Dict of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy" : accuracy_score(y_test, y_pred),
        "f1" : f1_score(y_test, y_pred, average="micro"),
        "precision" : precision_score(y_test, y_pred, average="micro"),
        "recall" : recall_score(y_test, y_pred, average="micro"),
    }

def main(model_config: dict):
    """
    Main training function. Loads data, trains models using specified
    classes/parameters/hyperparameters, evaluates models, and exports
    models to disk.
    
    :param model_config: Dict of model classes and corresponding parameters
                         and hyperparameters to use while training.
    """
    # Get datasets
    X_train, X_test, y_train, y_test = get_data()
    
    # For all models in config
    for name, config in model_config.items():

        # Build base model
        print(f"Building: {name}")
        model = build_model(model_class=name, model_params=config["params"])

        # Train model with hyperparameter tuning
        print(f"Training: {name}")
        model = train_model(model, config["hyperparameters"], X_train, y_train)
        print(f"Best parameters: {model.get_params()}")

        # Evaluate model
        print(f"Evaluating: {name}")
        metrics = evaluate_model(model, X_test, y_test)
        print(f"Evaluation metrics: {metrics}")

        # Export model to disk
        print(f"Saving: {name}")
        pickle.dump(model, open(f"{name}.pkl", 'wb'))
        print()