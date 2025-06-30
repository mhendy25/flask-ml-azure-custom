import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def train_and_save_model():
    """Train a housing price prediction model and save it"""
    logging.info("Loading dataset...")
    
    # Load California housing dataset (modern replacement for Boston dataset)
    housing = fetch_california_housing()
    
    # Use only a subset of features to match the original app's structure
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    
    # Select relevant features (similar to the original app's prediction script)
    X = X[['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude']]
    
    logging.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    logging.info("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    logging.info("Evaluating model...")
    score = model.score(X_test, y_test)
    logging.info(f"Model R² score: {score:.4f}")
    
    logging.info("Saving model...")
    joblib.dump(model, 'housing_model.joblib')
    logging.info("Model saved as housing_model.joblib")

if __name__ == "__main__":
    train_and_save_model()