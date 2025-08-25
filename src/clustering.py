"""
Clustering Analysis Module
=========================

This module implements various clustering algorithms for customer segmentation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import streamlit as st

class ClusteringAnalyzer:
    """
    Handles clustering analysis for customer segmentation.
    """
    
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.optimal_clusters = None
        self.cluster_labels = {}
        
    def find_optimal_clusters(self, scaled_data, max_clusters=10):
        """Find optimal number of clusters using multiple methods."""
        if scaled_data is None:
            st.error("No scaled data available. Please preprocess data first.")
            return None
        
        cluster_range = range(2, max_clusters + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, k in enumerate(cluster_range):
            status_text.text(f'Evaluating {k} clusters...')
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(scaled_data, cluster_labels))
            
            progress_bar.progress((i + 1) / len(cluster_range))
        
        status_text.text('Optimization complete!')
        
        # Find optimal clusters based on silhouette score
        optimal_silhouette = cluster_range[np.argmax(silhouette_scores)]
        optimal_calinski = cluster_range[np.argmax(calinski_scores)]
        
        # Store results
        self.optimization_results = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'optimal_silhouette': optimal_silhouette,
            'optimal_calinski': optimal_calinski
        }
        
        self.optimal_clusters = optimal_silhouette
        
        st.success(f"✅ Optimal clusters found: {self.optimal_clusters} (based on Silhouette Score)")
        
        return self.optimization_results
    
    def apply_kmeans(self, scaled_data, n_clusters=None):
        """Apply K-Means clustering."""
        if scaled_data is None:
            st.error("No scaled data available. Please preprocess data first.")
            return None
        
        if n_clusters is None:
            n_clusters = self.optimal_clusters or 5
        
        with st.spinner(f'Applying K-Means clustering with {n_clusters} clusters...'):
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = self.kmeans_model.fit_predict(scaled_data)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(scaled_data, kmeans_labels)
        calinski_score = calinski_harabasz_score(scaled_data, kmeans_labels)
        
        self.cluster_labels['kmeans'] = kmeans_labels
        
        results = {
            'labels': kmeans_labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_score,
            'inertia': self.kmeans_model.inertia_,
            'centers': self.kmeans_model.cluster_centers_
        }
        
        st.success(f"✅ K-Means clustering completed!")
        st.info(f"Silhouette Score: {silhouette_avg:.3f} | Calinski-Harabasz Score: {calinski_score:.3f}")
        
        return results
    
    def apply_dbscan(self, scaled_data, eps=0.5, min_samples=5):
        """Apply DBSCAN clustering."""
        if scaled_data is None:
            st.error("No scaled data available. Please preprocess data first.")
            return None
        
        with st.spinner(f'Applying DBSCAN clustering (eps={eps}, min_samples={min_samples})...'):
            self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = self.dbscan_model.fit_predict(scaled_data)
        
        # Calculate metrics
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        self.cluster_labels['dbscan'] = dbscan_labels
        
        results = {
            'labels': dbscan_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples
        }
        
        # Calculate silhouette score only if we have more than 1 cluster and non-noise points
        if n_clusters > 1:
            non_noise_mask = dbscan_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(scaled_data[non_noise_mask], 
                                                dbscan_labels[non_noise_mask])
                results['silhouette_score'] = silhouette_avg
        
        st.success(f"✅ DBSCAN clustering completed!")
        st.info(f"Clusters: {n_clusters} | Noise points: {n_noise}")
        
        return results
    
    def analyze_clusters(self, data, algorithm='kmeans'):
        """Analyze cluster characteristics."""
        # Normalize algorithm name
        algo_key = algorithm.lower().replace('-', '').replace(' ', '')
        
        if algo_key not in self.cluster_labels:
            st.error(f"No {algorithm} clustering results found. Please run clustering first.")
            return None
        
        cluster_labels = self.cluster_labels[algo_key]
        
        # Create consistent column name (use the format that actually gets created)
        if algo_key == 'kmeans':
            cluster_col = 'Kmeans_Cluster'  # Match what we see in the error
        elif algo_key == 'dbscan':
            cluster_col = 'DBSCAN_Cluster'
        else:
            cluster_col = f'{algorithm}_Cluster'
        
        # Add cluster labels to data
        analysis_data = data.copy()
        analysis_data[cluster_col] = cluster_labels
        
        # Calculate cluster statistics
        numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.endswith('_Cluster')]
        
        cluster_stats = analysis_data.groupby(cluster_col)[numeric_cols].agg(['mean', 'std', 'count'])
        
        # Calculate spending analysis if available
        spending_analysis = None
        if 'Spending Score (1-100)' in analysis_data.columns:
            spending_analysis = analysis_data.groupby(cluster_col)['Spending Score (1-100)'].agg(['mean', 'std', 'min', 'max', 'count'])
        
        results = {
            'data_with_clusters': analysis_data,
            'cluster_stats': cluster_stats,
            'spending_analysis': spending_analysis,
            'cluster_distribution': analysis_data[cluster_col].value_counts().sort_index()
        }
        
        return results
    
    def get_cluster_profiles(self, data, algorithm='kmeans'):
        """Generate customer profiles for each cluster."""
        # Normalize algorithm name
        algo_key = algorithm.lower().replace('-', '').replace(' ', '')
        
        if algo_key not in self.cluster_labels:
            return None
        
        cluster_labels = self.cluster_labels[algo_key]
        
        # Create consistent column name (use the format that actually gets created)
        if algo_key == 'kmeans':
            cluster_col = 'Kmeans_Cluster'  # Match what we see in the error
        elif algo_key == 'dbscan':
            cluster_col = 'DBSCAN_Cluster'
        else:
            cluster_col = f'{algorithm}_Cluster'
        
        analysis_data = data.copy()
        analysis_data[cluster_col] = cluster_labels
        
        profiles = []
        
        for cluster in sorted(analysis_data[cluster_col].unique()):
            if cluster == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_data = analysis_data[analysis_data[cluster_col] == cluster]
            
            profile = {
                'cluster': cluster,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(analysis_data) * 100
            }
            
            # Add feature statistics
            if 'Age' in cluster_data.columns:
                profile['avg_age'] = cluster_data['Age'].mean()
                profile['age_std'] = cluster_data['Age'].std()
            
            if 'Annual Income (k$)' in cluster_data.columns:
                profile['avg_income'] = cluster_data['Annual Income (k$)'].mean()
                profile['income_std'] = cluster_data['Annual Income (k$)'].std()
            
            if 'Spending Score (1-100)' in cluster_data.columns:
                profile['avg_spending'] = cluster_data['Spending Score (1-100)'].mean()
                profile['spending_std'] = cluster_data['Spending Score (1-100)'].std()
            
            if 'Gender' in cluster_data.columns:
                profile['gender_dist'] = cluster_data['Gender'].value_counts().to_dict()
            
            # Generate profile characterization
            if 'avg_income' in profile and 'avg_spending' in profile:
                avg_income = profile['avg_income']
                avg_spending = profile['avg_spending']
                
                if avg_income > 70 and avg_spending > 70:
                    profile['type'] = "💎 HIGH VALUE"
                    profile['description'] = "High income, high spending - Premium customers"
                elif avg_income > 70 and avg_spending < 40:
                    profile['type'] = "💼 CONSERVATIVE"
                    profile['description'] = "High income, low spending - Potential for upselling"
                elif avg_income < 40 and avg_spending > 70:
                    profile['type'] = "🎯 BUDGET SPENDERS"
                    profile['description'] = "Low income, high spending - Price-sensitive loyal customers"
                elif avg_income < 40 and avg_spending < 40:
                    profile['type'] = "📉 LOW ENGAGEMENT"
                    profile['description'] = "Low income, low spending - Need retention strategies"
                else:
                    profile['type'] = "⚖️ BALANCED"
                    profile['description'] = "Moderate income and spending - Core customer base"
            
            profiles.append(profile)
        
        return profiles
