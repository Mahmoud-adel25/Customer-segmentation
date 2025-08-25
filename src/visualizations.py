"""
Visualization Module
===================

This module handles all visualization components for the customer segmentation analysis.
"""

# Matplotlib and Seaborn removed to avoid extra dependency
# All charts use Plotly for interactive visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
import streamlit as st

# Global Plotly template: dark backgrounds to match app theme
pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"].layout.update(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font=dict(color="#E5E7EB")
)

# Plot styling handled via Plotly theme settings per figure

class Visualizer:
    """
    Handles all visualizations for customer segmentation analysis.
    """
    
    def __init__(self):
        # Enhanced color palettes for better visual appeal
        self.colors = px.colors.qualitative.Set1  # More vibrant colors
        self.gradient_colors = [
            '#FF6B6B',  # Coral Red
            '#4ECDC4',  # Turquoise
            '#45B7D1',  # Sky Blue
            '#96CEB4',  # Mint Green
            '#FFEAA7',  # Warm Yellow
            '#DDA0DD',  # Plum
            '#98D8C8',  # Seafoam
            '#F7DC6F',  # Golden Yellow
            '#BB8FCE',  # Lavender
            '#85C1E9'   # Light Blue
        ]
        self.modern_colors = [
            '#6C5CE7',  # Purple
            '#00B894',  # Green
            '#E17055',  # Orange
            '#0984E3',  # Blue
            '#FDCB6E',  # Yellow
            '#E84393',  # Pink
            '#00CEC9',  # Cyan
            '#A29BFE',  # Light Purple
            '#FD79A8',  # Light Pink
            '#81ECEC'   # Light Cyan
        ]
    
    def plot_data_exploration(self, data):
        """Create comprehensive data exploration plots with enhanced styling."""
        if data is None:
            st.error("❌ No data available for visualization.")
            return
        
        # Debug: Show data info
        st.info(f"🔍 **Data shape:** {data.shape}")
        st.info(f"🔍 **Data columns:** {list(data.columns)}")
        
        st.subheader("📊 Data Distribution Analysis")
        
        # Create subplots for different visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution with enhanced styling
            if 'Age' in data.columns:
                st.write("📊 Creating Age distribution plot...")
                fig_age = px.histogram(
                    data, x='Age', nbins=20, 
                    title='👥 Age Distribution',
                    color_discrete_sequence=[self.gradient_colors[0]]
                )
                fig_age.update_layout(
                    height=450,
                    title=dict(font=dict(size=18, color='#E5E7EB'), x=0.5),
                    plot_bgcolor='#0F172A',
                    paper_bgcolor='#0F172A',
                    xaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB')),
                    yaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB'))
                )
                fig_age.update_traces(marker=dict(line=dict(width=1, color='white')))
                st.plotly_chart(fig_age, use_container_width=True, theme=None)
                st.success("✅ Age distribution plot created!")
            
            # Income distribution with enhanced styling
            if 'Annual Income (k$)' in data.columns:
                st.write("💰 Creating Income distribution plot...")
                fig_income = px.histogram(
                    data, x='Annual Income (k$)', nbins=20,
                    title='💰 Annual Income Distribution',
                    color_discrete_sequence=[self.gradient_colors[1]]
                )
                fig_income.update_layout(
                    height=450,
                    title=dict(font=dict(size=18, color='#E5E7EB'), x=0.5),
                    plot_bgcolor='#0F172A',
                    paper_bgcolor='#0F172A',
                    xaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB')),
                    yaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB'))
                )
                fig_income.update_traces(marker=dict(line=dict(width=1, color='white')))
                st.plotly_chart(fig_income, use_container_width=True, theme=None)
                st.success("✅ Income distribution plot created!")
        
        with col2:
            # Spending Score distribution with enhanced styling
            if 'Spending Score (1-100)' in data.columns:
                st.write("🛍️ Creating Spending Score distribution plot...")
                fig_spending = px.histogram(
                    data, x='Spending Score (1-100)', nbins=20,
                    title='🛍️ Spending Score Distribution',
                    color_discrete_sequence=[self.gradient_colors[2]]
                )
                fig_spending.update_layout(
                    height=450,
                    title=dict(font=dict(size=18, color='#E5E7EB'), x=0.5),
                    plot_bgcolor='#0F172A',
                    paper_bgcolor='#0F172A',
                    xaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB')),
                    yaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB'))
                )
                fig_spending.update_traces(marker=dict(line=dict(width=1, color='white')))
                st.plotly_chart(fig_spending, use_container_width=True, theme=None)
                st.success("✅ Spending Score distribution plot created!")
            
            # Gender distribution with enhanced styling
            if 'Gender' in data.columns:
                gender_counts = data['Gender'].value_counts()
                fig_gender = px.pie(
                    values=gender_counts.values, 
                    names=gender_counts.index,
                    title='👫 Gender Distribution',
                    color_discrete_sequence=self.modern_colors[:len(gender_counts)]
                )
                fig_gender.update_layout(
                    height=450,
                    title=dict(font=dict(size=18, color='#E5E7EB'), x=0.5),
                    plot_bgcolor='#0F172A',
                    paper_bgcolor='#0F172A'
                )
                fig_gender.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=14,
                    marker=dict(line=dict(color='white', width=2))
                )
                st.plotly_chart(fig_gender, use_container_width=True)
        
        # Enhanced correlation analysis
        st.subheader("🔗 Feature Correlations")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=True, 
                title='🔗 Feature Correlation Matrix',
                color_continuous_scale='RdYlBu',
                aspect='auto'
            )
            fig_corr.update_layout(
                height=500,
                title=dict(font=dict(size=18, color='#E5E7EB'), x=0.5),
                plot_bgcolor='#0F172A',
                paper_bgcolor='#0F172A',
                font=dict(size=12, color='#E5E7EB')
            )
            fig_corr.update_traces(
                textfont=dict(size=12, color='#E5E7EB'),
                hoverongaps=False
            )
            st.plotly_chart(fig_corr, theme=None, use_container_width=True)
        
        # Enhanced scatter plots
        st.subheader("🔍 Feature Relationships")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Annual Income (k$)' in data.columns and 'Spending Score (1-100)' in data.columns:
                fig_scatter1 = px.scatter(
                    data, x='Annual Income (k$)', y='Spending Score (1-100)',
                    title='💰 Income vs Spending Score',
                    hover_data=['Age'] if 'Age' in data.columns else None,
                    color_discrete_sequence=[self.modern_colors[3]]
                )
                fig_scatter1.update_layout(
                    height=450,
                    title=dict(font=dict(size=18, color='#E5E7EB'), x=0.5),
                    plot_bgcolor='#0F172A',
                    paper_bgcolor='#0F172A',
                    xaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB')),
                    yaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB'))
                )
                fig_scatter1.update_traces(
                    marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white'))
                )
                st.plotly_chart(fig_scatter1, use_container_width=True)
        
        with col2:
            if 'Age' in data.columns and 'Spending Score (1-100)' in data.columns:
                fig_scatter2 = px.scatter(
                    data, x='Age', y='Spending Score (1-100)',
                    title='👥 Age vs Spending Score',
                    hover_data=['Annual Income (k$)'] if 'Annual Income (k$)' in data.columns else None,
                    color_discrete_sequence=[self.modern_colors[4]]
                )
                fig_scatter2.update_layout(
                    height=450,
                    title=dict(font=dict(size=18, color='#E5E7EB'), x=0.5),
                    plot_bgcolor='#0F172A',
                    paper_bgcolor='#0F172A',
                    xaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB')),
                    yaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB'))
                )
                fig_scatter2.update_traces(
                    marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white'))
                )
                st.plotly_chart(fig_scatter2, use_container_width=True)
    
    def plot_optimization_results(self, results):
        """Plot cluster optimization results."""
        if results is None:
            st.error("No optimization results available.")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Elbow Method', 'Silhouette Score', 'Calinski-Harabasz Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        cluster_range = results['cluster_range']
        
        # Elbow method
        fig.add_trace(
            go.Scatter(x=cluster_range, y=results['inertias'], 
                      mode='lines+markers', name='Inertia',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Silhouette score
        fig.add_trace(
            go.Scatter(x=cluster_range, y=results['silhouette_scores'], 
                      mode='lines+markers', name='Silhouette Score',
                      line=dict(color='red')),
            row=1, col=2
        )
        
        # Calinski-Harabasz score
        fig.add_trace(
            go.Scatter(x=cluster_range, y=results['calinski_scores'], 
                      mode='lines+markers', name='Calinski-Harabasz Score',
                      line=dict(color='green')),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Cluster Optimization Results",
            height=400,
            showlegend=False,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="#E5E7EB")
        )
        
        fig.update_xaxes(title_text="Number of Clusters")
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        fig.update_yaxes(title_text="Calinski-Harabasz Score", row=1, col=3)
        
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
        # Display optimal results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimal Clusters (Silhouette)", results['optimal_silhouette'])
        with col2:
            st.metric("Optimal Clusters (Calinski-Harabasz)", results['optimal_calinski'])
        with col3:
            st.metric("Recommended", results['optimal_silhouette'])
    
    def plot_clusters(self, data, cluster_labels, algorithm='K-Means', scaler=None, centers=None):
        """Plot cluster visualizations."""
        if data is None or cluster_labels is None:
            st.error("No data or cluster labels available for visualization.")
            return
        
        # Prepare data with clusters
        plot_data = data.copy()
        plot_data['Cluster'] = cluster_labels
        
        # Main clustering visualization
        st.subheader(f"🎯 {algorithm} Clustering Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Annual Income (k$)' in data.columns and 'Spending Score (1-100)' in data.columns:
                fig_main = px.scatter(plot_data, 
                                    x='Annual Income (k$)', 
                                    y='Spending Score (1-100)',
                                    color='Cluster',
                                    title=f'{algorithm}: Income vs Spending Score',
                                    hover_data=['Age'] if 'Age' in data.columns else None,
                                    color_discrete_sequence=self.colors)
                
                # Add cluster centers if available
                if centers is not None and scaler is not None:
                    centers_original = scaler.inverse_transform(centers)
                    centers_df = pd.DataFrame(centers_original, 
                                            columns=['Annual Income (k$)', 'Spending Score (1-100)'])
                    centers_df['Cluster'] = range(len(centers_df))
                    
                    fig_main.add_scatter(x=centers_df['Annual Income (k$)'], 
                                       y=centers_df['Spending Score (1-100)'],
                                       mode='markers',
                                       marker=dict(symbol='x', size=15, color='red', line=dict(width=2)),
                                       name='Centers',
                                       showlegend=True)
                
                fig_main.update_layout(
                    height=500,
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#0F172A",
                    font=dict(color="#E5E7EB"),
                    xaxis=dict(gridcolor="rgba(229,231,235,0.12)"),
                    yaxis=dict(gridcolor="rgba(229,231,235,0.12)")
                )
                st.plotly_chart(fig_main, theme=None, use_container_width=True)
        
        with col2:
            if 'Age' in data.columns and 'Spending Score (1-100)' in data.columns:
                fig_age = px.scatter(plot_data, 
                                   x='Age', 
                                   y='Spending Score (1-100)',
                                   color='Cluster',
                                   title=f'{algorithm}: Age vs Spending Score',
                                   color_discrete_sequence=self.colors)
                fig_age.update_layout(
                    height=500,
                    paper_bgcolor="#0F172A",
                    plot_bgcolor="#0F172A",
                    font=dict(color="#E5E7EB"),
                    xaxis=dict(gridcolor="rgba(229,231,235,0.12)"),
                    yaxis=dict(gridcolor="rgba(229,231,235,0.12)")
                )
                st.plotly_chart(fig_age, theme=None, use_container_width=True)
        
        # Enhanced cluster distribution
        st.subheader("📊 Cluster Distribution")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        fig_dist = px.bar(
            x=cluster_counts.index, y=cluster_counts.values,
            title='📊 Number of Customers per Cluster',
            labels={'x': 'Cluster', 'y': 'Number of Customers'},
            color=cluster_counts.values,
            color_continuous_scale='Turbo'
        )
        fig_dist.update_layout(
            height=450,
            title=dict(font=dict(size=18, color='#E5E7EB'), x=0.5),
            plot_bgcolor='#0F172A',
            paper_bgcolor='#0F172A',
            xaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB')),
            yaxis=dict(gridcolor='rgba(229,231,235,0.12)', title_font=dict(size=14, color='#E5E7EB'))
        )
        fig_dist.update_traces(
            marker=dict(line=dict(width=1, color='white'))
        )
        st.plotly_chart(fig_dist, theme=None, use_container_width=True)
    
    def plot_cluster_analysis(self, analysis_results, algorithm='K-Means'):
        """Plot detailed cluster analysis with enhanced visualizations."""
        if analysis_results is None:
            st.error("❌ No analysis results available.")
            return
        
        try:
            data_with_clusters = analysis_results['data_with_clusters']
            spending_analysis = analysis_results['spending_analysis']
            
            # COMPLETELY REWRITTEN: Find cluster column with bulletproof detection
            available_columns = list(data_with_clusters.columns)
            st.info(f"🔍 **Available columns in data:** {available_columns}")
            
            # Find ANY column that contains 'cluster' (case insensitive)
            cluster_columns = [col for col in available_columns if 'cluster' in col.lower()]
            st.info(f"🎯 **Found cluster columns:** {cluster_columns}")
            
            if not cluster_columns:
                st.error("❌ No cluster column found in the data!")
                st.write("Available columns:", available_columns)
                st.write("Please ensure clustering has been performed first.")
                return
            
            # Use the first cluster column found
            cluster_col = cluster_columns[0]
            st.success(f"✅ **Using cluster column:** `{cluster_col}`")
            
            # EXTRA SAFETY: Ensure the column actually exists before proceeding
            if cluster_col not in data_with_clusters.columns:
                st.error(f"❌ Column `{cluster_col}` not found in data!")
                st.write("This should not happen. Please report this bug.")
                return
            
            # Create a beautiful header with metrics
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin: 2rem 0;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            ">
                <h2 style="margin: 0; font-size: 2.5rem; font-weight: 700;">📈 {algorithm} Cluster Analysis</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Interactive Cluster Visualization & Analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick stats
            num_clusters = len(data_with_clusters[cluster_col].unique())
            total_customers = len(data_with_clusters)
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("🎯 Total Clusters", num_clusters)
            with metric_col2:
                st.metric("👥 Total Customers", total_customers)
            with metric_col3:
                avg_cluster_size = total_customers / num_clusters
                st.metric("📊 Avg Cluster Size", f"{avg_cluster_size:.0f}")
            with metric_col4:
                if 'Spending Score (1-100)' in data_with_clusters.columns:
                    avg_spending = data_with_clusters['Spending Score (1-100)'].mean()
                    st.metric("💰 Avg Spending", f"{avg_spending:.1f}")
            
            st.markdown("---")
            
            # Enhanced Box plots with better styling
            st.subheader("📊 Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Spending Score (1-100)' in data_with_clusters.columns:
                    # Convert cluster column to string to ensure proper categorical handling
                    plot_data = data_with_clusters.copy()
                    plot_data[cluster_col] = plot_data[cluster_col].astype(str)
                    
                    # DEBUG: Show exactly what we're passing to plotly
                    st.write(f"🔍 **DEBUG - About to create box plot with:**")
                    st.write(f"- x column: `{cluster_col}`")
                    st.write(f"- Columns in plot_data: {list(plot_data.columns)}")
                    st.write(f"- First few rows of plot_data:")
                    st.dataframe(plot_data.head(3))
                    
                    fig_spending_box = px.box(
                        plot_data, 
                        x=cluster_col, 
                        y='Spending Score (1-100)',
                        title='💰 Spending Score Distribution by Cluster',
                        color=cluster_col,
                        color_discrete_sequence=self.modern_colors
                    )
                    
                    # Enhanced styling for maximum visibility
                    fig_spending_box.update_layout(
                        height=600,
                        title=dict(
                            text='💰 Spending Score Distribution by Cluster',
                            font=dict(size=20, color='#E5E7EB'),
                            x=0.5,
                            y=0.95
                        ),
                        plot_bgcolor='#0F172A',
                        paper_bgcolor='#0F172A',
                        font=dict(size=14, family="Arial, sans-serif", color='#E5E7EB'),
                        xaxis=dict(
                            title=dict(text='Cluster', font=dict(size=16, color='#E5E7EB')),
                            tickfont=dict(size=14, color='#E5E7EB'),
                            gridcolor='rgba(229,231,235,0.12)',
                            gridwidth=1,
                            showgrid=True
                        ),
                        yaxis=dict(
                            title=dict(text='Spending Score', font=dict(size=16, color='#E5E7EB')),
                            tickfont=dict(size=14, color='#E5E7EB'),
                            gridcolor='rgba(229,231,235,0.12)',
                            gridwidth=1,
                            showgrid=True
                        ),
                        showlegend=False,
                        margin=dict(t=80, b=60, l=60, r=40)
                    )
                    
                    fig_spending_box.update_traces(
                        marker=dict(size=6, opacity=0.8),
                        line=dict(width=3),
                        fillcolor='rgba(0,0,0,0)',
                        boxpoints='outliers'
                    )
                    
                    st.plotly_chart(fig_spending_box, theme=None, use_container_width=True)
            
            with col2:
                if 'Annual Income (k$)' in data_with_clusters.columns:
                    # Convert cluster column to string to ensure proper categorical handling
                    plot_data = data_with_clusters.copy()
                    plot_data[cluster_col] = plot_data[cluster_col].astype(str)
                    
                    fig_income_box = px.box(
                        plot_data, 
                        x=cluster_col, 
                        y='Annual Income (k$)',
                        title='💵 Income Distribution by Cluster',
                        color=cluster_col,
                        color_discrete_sequence=self.modern_colors
                    )
                    
                    # Enhanced styling for maximum visibility
                    fig_income_box.update_layout(
                        height=600,
                        title=dict(
                            text='💵 Annual Income Distribution by Cluster',
                            font=dict(size=20, color='#E5E7EB'),
                            x=0.5,
                            y=0.95
                        ),
                        plot_bgcolor='#0F172A',
                        paper_bgcolor='#0F172A',
                        font=dict(size=14, family="Arial, sans-serif", color='#E5E7EB'),
                        xaxis=dict(
                            title=dict(text='Cluster', font=dict(size=16, color='#E5E7EB')),
                            tickfont=dict(size=14, color='#E5E7EB'),
                            gridcolor='rgba(229,231,235,0.12)',
                            gridwidth=1,
                            showgrid=True
                        ),
                        yaxis=dict(
                            title=dict(text='Annual Income (k$)', font=dict(size=16, color='#E5E7EB')),
                            tickfont=dict(size=14, color='#E5E7EB'),
                            gridcolor='rgba(229,231,235,0.12)',
                            gridwidth=1,
                            showgrid=True
                        ),
                        showlegend=False,
                        margin=dict(t=80, b=60, l=60, r=40)
                    )
                    
                    fig_income_box.update_traces(
                        marker=dict(size=6, opacity=0.8),
                        line=dict(width=3),
                        fillcolor='rgba(0,0,0,0)',
                        boxpoints='outliers'
                    )
                    
                    st.plotly_chart(fig_income_box, theme=None, use_container_width=True)
        
            # Average spending per cluster with stunning visualization
            if spending_analysis is not None:
                st.markdown("---")
                
                # Beautiful section header
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    color: white;
                    text-align: center;
                    margin: 2rem 0 1rem 0;
                    box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);
                ">
                    <h3 style="margin: 0; font-size: 1.8rem; font-weight: 600;">💰 Average Spending Analysis</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create stunning bar chart with enhanced colors
                fig_avg_spending = px.bar(
                    x=spending_analysis.index.astype(str), 
                    y=spending_analysis['mean'],
                    title='📊 Average Spending Score by Cluster',
                    labels={'x': 'Cluster', 'y': 'Average Spending Score'},
                    error_y=spending_analysis['std'],
                    color=spending_analysis['mean'],
                    color_continuous_scale='Viridis'
                )
                
                # Ultra-enhanced styling
                fig_avg_spending.update_layout(
                     height=650,
                     title=dict(
                         text='📊 Average Spending Score by Cluster',
                         font=dict(size=24, color='#E5E7EB', family="Arial Black"),
                         x=0.5,
                         y=0.95
                     ),
                     plot_bgcolor='#0F172A',
                     paper_bgcolor='#0F172A',
                     font=dict(size=16, family="Arial, sans-serif", color='#E5E7EB'),
                     xaxis=dict(
                         title=dict(text='Cluster', font=dict(size=18, color='#E5E7EB')),
                         tickfont=dict(size=16, color='#E5E7EB'),
                         gridcolor='rgba(229,231,235,0.12)',
                         gridwidth=1,
                         showgrid=True,
                         zeroline=False
                     ),
                     yaxis=dict(
                         title=dict(text='Average Spending Score', font=dict(size=18, color='#E5E7EB')),
                         tickfont=dict(size=16, color='#E5E7EB'),
                         gridcolor='rgba(229,231,235,0.12)',
                         gridwidth=1,
                         showgrid=True,
                         zeroline=False
                     ),
                     showlegend=False,
                     margin=dict(t=100, b=80, l=80, r=80)
                 )
                
                # Add stylish value labels on bars
                for i, (cluster, value) in enumerate(zip(spending_analysis.index, spending_analysis['mean'])):
                    fig_avg_spending.add_annotation(
                        x=str(cluster), 
                        y=value + spending_analysis.loc[cluster, 'std'] + 5,
                        text=f'<b>{value:.1f}</b>',
                        showarrow=False,
                        font=dict(size=16, color='white', family="Arial Black"),
                        bgcolor='rgba(44, 62, 80, 0.9)',
                        bordercolor='rgba(44, 62, 80, 1)',
                        borderwidth=2,
                        borderpad=8
                    )
                
                # Enhance the bars themselves
                fig_avg_spending.update_traces(
                    marker=dict(
                        line=dict(width=2, color='rgba(44, 62, 80, 0.8)'),
                        opacity=0.9
                    ),
                    width=0.6
                )
                
                st.plotly_chart(fig_avg_spending, theme=None, use_container_width=True)
                
                # Beautiful cluster insights table
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    color: white;
                    text-align: center;
                    margin: 2rem 0 1rem 0;
                    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
                ">
                    <h3 style="margin: 0; font-size: 1.8rem; font-weight: 600;">📋 Detailed Cluster Statistics</h3>
                </div>
                """, unsafe_allow_html=True)
                
                summary_df = spending_analysis.round(2)
                summary_df.columns = ['🎯 Avg Spending', '📊 Std Dev', '📉 Min', '📈 Max', '👥 Count']
                
                # Create a Plotly table instead of using background_gradient
                fig_table = go.Figure(data=[go.Table(
                    header=dict(
                        values=list(summary_df.columns),
                        fill_color='#1F2937',
                        font=dict(color='#E5E7EB', size=14, family='Inter'),
                        align='center',
                        height=40
                    ),
                    cells=dict(
                        values=[summary_df[col] for col in summary_df.columns],
                        fill_color='#0F172A',
                        font=dict(color='#E5E7EB', size=12, family='Inter'),
                        align='center',
                        height=35,
                        format=[None, '.2f', '.2f', '.2f', '.2f', '.0f']
                    )
                )])
                
                fig_table.update_layout(
                     height=300,
                     title=dict(
                         text='📊 Cluster Spending Analysis',
                         font=dict(size=18, color='#E5E7EB', family='Inter'),
                         x=0.5
                     ),
                     plot_bgcolor='#0F172A',
                     paper_bgcolor='#0F172A',
                     margin=dict(t=60, b=20, l=20, r=20)
                 )
                st.plotly_chart(fig_table, use_container_width=True, theme=None)
        
        except Exception as e:
            st.error(f"❌ Error in cluster analysis visualization: {str(e)}")
            st.write("Please try the 'Clear Session' button in the sidebar and run clustering again.")
    
    def plot_comparison(self, data, kmeans_labels, dbscan_labels):
        """Plot comparison between K-Means and DBSCAN."""
        st.subheader("🔄 Algorithm Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # K-Means
            plot_data_kmeans = data.copy()
            plot_data_kmeans['Cluster'] = kmeans_labels
            
            fig_kmeans = px.scatter(plot_data_kmeans, 
                                  x='Annual Income (k$)', 
                                  y='Spending Score (1-100)',
                                  color='Cluster',
                                  title='K-Means Clustering',
                                  color_discrete_sequence=self.colors)
            fig_kmeans.update_layout(
                height=400,
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                font=dict(color="#E5E7EB")
            )
            st.plotly_chart(fig_kmeans, theme=None, use_container_width=True)
        
        with col2:
            # DBSCAN
            plot_data_dbscan = data.copy()
            plot_data_dbscan['Cluster'] = dbscan_labels
            plot_data_dbscan['Cluster'] = plot_data_dbscan['Cluster'].astype(str)
            plot_data_dbscan.loc[plot_data_dbscan['Cluster'] == '-1', 'Cluster'] = 'Noise'
            
            fig_dbscan = px.scatter(plot_data_dbscan, 
                                  x='Annual Income (k$)', 
                                  y='Spending Score (1-100)',
                                  color='Cluster',
                                  title='DBSCAN Clustering',
                                  color_discrete_sequence=self.colors)
            fig_dbscan.update_layout(
                height=400,
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                font=dict(color="#E5E7EB")
            )
            st.plotly_chart(fig_dbscan, theme=None, use_container_width=True)
        
        # Comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            kmeans_clusters = len(set(kmeans_labels))
            st.metric("K-Means Clusters", kmeans_clusters)
        
        with col2:
            dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            st.metric("DBSCAN Clusters", dbscan_clusters)
        
        with col3:
            noise_points = list(dbscan_labels).count(-1)
            st.metric("DBSCAN Noise Points", noise_points)
        
        with col4:
            noise_percentage = (noise_points / len(dbscan_labels)) * 100
            st.metric("Noise Percentage", f"{noise_percentage:.1f}%")
