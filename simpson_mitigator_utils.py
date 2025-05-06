import numpy as np
import pandas as pd

def preprocess_data(data):
    """
    Preprocess dataset by handling missing values and normalizing continuous variables.
    
    Parameters:
    - data: pandas DataFrame
    
    Returns:
    - pandas DataFrame, preprocessed
    """
    data = data.copy()
    # Handle missing values
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].fillna(data[col].mean())
    for col in data.select_dtypes(include=['object', 'category']).columns:
        data[col] = data[col].fillna(data[col].mode()[0]).astype('category')
    
    # Normalize continuous variables
    for col in data.select_dtypes(include=[np.number]).columns:
        if data[col].std() > 0:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    
    return data

def compute_fairness_metrics(data, y_pred, protected_attribute):
    """
    Compute fairness metrics: Demographic Parity Difference (DPD) and Equal Opportunity Difference (EOD).
    
    Parameters:
    - data: pandas DataFrame
    - y_pred: array, predicted labels
    - protected_attribute: str, column name for protected attribute
    
    Returns:
    - dict, containing DPD and EOD
    """
    groups = data[protected_attribute].unique()
    if len(groups) != 2:
        raise ValueError("Fairness metrics require exactly two groups")
    
    group0, group1 = groups
    mask0 = data[protected_attribute] == group0
    mask1 = data[protected_attribute] == group1
    
    # Demographic Parity Difference
    dp0 = np.mean(y_pred[mask0])
    dp1 = np.mean(y_pred[mask1])
    dpd = abs(dp0 - dp1)
    
    # Equal Opportunity Difference (assumes binary outcome)
    eod0 = np.mean(y_pred[mask0 & (data['admitted'] == 1)])
    eod1 = np.mean(y_pred[mask1 & (data['admitted'] == 1)])
    eod = abs(eod0 - eod1) if not (np.isnan(eod0) or np.isnan(eod1)) else 0.0
    
    return {'DPD': dpd, 'EOD': eod}