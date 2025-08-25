"""
Data Generation Utilities
========================

Utility functions for generating sample datasets.
"""

import pandas as pd
import numpy as np

def create_sample_mall_customers(n_customers=200, random_seed=42):
    """
    Create a realistic sample Mall Customers dataset.
    
    Parameters:
    -----------
    n_customers : int, default=200
        Number of customers to generate
    random_seed : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Generated customer dataset
    """
    np.random.seed(random_seed)
    
    customer_ids = range(1, n_customers + 1)
    
    # Gender distribution (approximately 56% Female, 44% Male)
    genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.44, 0.56])
    
    # Age distribution (mean ~39, std ~14)
    ages = np.random.normal(38.85, 13.97, n_customers).astype(int)
    ages = np.clip(ages, 18, 70)
    
    # Create realistic income distribution (mean ~61k, std ~26k)
    annual_incomes = np.random.normal(60.56, 26.26, n_customers)
    annual_incomes = np.clip(annual_incomes, 15, 137)
    
    # Create spending scores with realistic patterns
    base_spending = np.random.normal(50, 25, n_customers)
    
    # Add some income correlation
    income_normalized = (annual_incomes - annual_incomes.min()) / (annual_incomes.max() - annual_incomes.min())
    income_effect = (income_normalized - 0.5) * 30
    
    # Add age effect (younger people might spend more)
    age_normalized = (ages - ages.min()) / (ages.max() - ages.min())
    age_effect = np.where(age_normalized < 0.3, 10,
                         np.where(age_normalized > 0.7, -5, 0))
    
    # Gender effect (slight difference in spending patterns)
    gender_effect = np.where(genders == 'Female', 3, -3)
    
    spending_scores = (base_spending + 
                      income_effect * 0.6 + 
                      age_effect + 
                      gender_effect +
                      np.random.normal(0, 10, n_customers))
    spending_scores = np.clip(spending_scores, 1, 100)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Annual Income (k$)': annual_incomes.round().astype(int),
        'Spending Score (1-100)': spending_scores.round().astype(int)
    })
    
    return sample_data
