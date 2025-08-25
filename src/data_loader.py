"""
Data Loading and Preprocessing Module
====================================

This module handles data loading, preprocessing, and validation for customer segmentation.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import streamlit as st

class DataLoader:
    """
    Handles data loading and preprocessing for customer segmentation analysis.
    """
    
    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_sample_dataset(self, n_customers=200):
        """Create a realistic sample Mall Customers dataset."""
        np.random.seed(42)
        
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
        
        # Add age effect
        age_normalized = (ages - ages.min()) / (ages.max() - ages.min())
        age_effect = np.where(age_normalized < 0.3, 10,
                             np.where(age_normalized > 0.7, -5, 0))
        
        spending_scores = base_spending + income_effect * 0.6 + age_effect + np.random.normal(0, 10, n_customers)
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
    
    def load_data(self, file_path=None):
        """Load customer data from file or create sample data."""
        # Check for default dataset location first
        default_path = os.path.join("data", "Mall_Customers.csv")
        
        if file_path and os.path.exists(file_path):
            try:
                self.data = pd.read_csv(file_path)
                st.success(f"✅ Data loaded successfully from {file_path}")
                return self.data
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return None
        elif os.path.exists(default_path):
            try:
                self.data = pd.read_csv(default_path)
                st.success(f"✅ Mall Customers dataset loaded from {default_path}")
                return self.data
            except Exception as e:
                st.error(f"Error loading default dataset: {e}")
                return None
        else:
            # Create sample data
            self.data = self.create_sample_dataset()
            st.info("📊 Using generated sample dataset (Mall Customer simulation)")
            # Save the sample data for future use
            try:
                os.makedirs("data", exist_ok=True)
                self.data.to_csv(default_path, index=False)
                st.info(f"💾 Sample dataset saved to {default_path}")
            except Exception as e:
                st.warning(f"Could not save sample dataset: {e}")
            return self.data
    
    def get_data_info(self):
        """Get comprehensive data information."""
        if self.data is None:
            return None
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'statistics': self.data.describe().to_dict()
        }
        return info
    
    def preprocess_data(self, features=None):
        """Preprocess and scale data for clustering."""
        if self.data is None:
            st.error("No data loaded. Please load data first.")
            return None
        
        # Default features for clustering
        if features is None:
            features = ['Annual Income (k$)', 'Spending Score (1-100)']
        
        # Check if features exist in data
        available_features = [f for f in features if f in self.data.columns]
        if not available_features:
            st.error(f"None of the specified features {features} found in data.")
            return None
        
        # Extract features for clustering
        X = self.data[available_features].copy()
        
        # Handle missing values if any
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.mean())
            st.warning("Missing values filled with mean values.")
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(X)
        self.feature_names = available_features
        
        st.success(f"✅ Data preprocessed successfully using features: {available_features}")
        return self.scaled_data
    
    def get_feature_data(self):
        """Get the original feature data."""
        if self.data is None or self.feature_names is None:
            return None
        return self.data[self.feature_names]
