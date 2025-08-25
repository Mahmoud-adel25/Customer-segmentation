"""
Customer Segmentation Streamlit App
==================================

A comprehensive web application for customer segmentation analysis using
K-Means and DBSCAN clustering algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.clustering import ClusteringAnalyzer
from src.visualizations import Visualizer

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Modern Dark Mode Compatible CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for Dark Mode Support */
    /* :root {
        --bg-primary: #0F172A;       /* slate-900 */
        --bg-secondary: #111827;     /* gray-900 */
        --bg-tertiary: #1F2937;      /* gray-800 */
        --text-primary: #E5E7EB;     /* gray-200 */
        --text-secondary: #CBD5E1;   /* slate-300 */
        --text-tertiary: #94A3B8;    /* slate-400 */
        --border-color: #374151;     /* gray-700 */
        --accent-primary: #818CF8;   /* indigo-300 */
        --accent-secondary: #A78BFA; /* violet-300 */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6);
    } */
    
    /* Dark mode support disabled intentionally */
    
    /* Base styling */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Apply CSS variables to Streamlit elements */
    .stApp { background-color: #0F172A; color: #E5E7EB; }
    
    /* Headers */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 800;
        text-align: center;
        margin-bottom: 3rem;
        background: linear-gradient(135deg, #818CF8 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #E5E7EB;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #374151;
        position: relative;
    }
    
    .sub-header::after {
        content: '';
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: linear-gradient(135deg, #818CF8, #A78BFA);
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #111827;
        padding: 8px;
        border-radius: 16px;
        border: 1px solid #374151;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 20px;
        background: transparent;
        border-radius: 12px;
        color: #CBD5E1;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        border: none;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #1F2937;
        color: #E5E7EB;
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #818CF8 0%, #A78BFA 100%);
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        transform: translateY(-1px);
    }
    
    /* Cards and containers */
    .metric-card {
        background: #0F172A;
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(135deg, #818CF8, #A78BFA);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.6);
        border-color: #818CF8;
    }
    
    .insight-box {
        background: #111827;
        border: 1px solid #818CF8;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
        position: relative;
    }
    
    .insight-box::before {
        content: '';
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(135deg, #818CF8, #A78BFA);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #111827;
        border-right: 1px solid #374151;
    }
    
    /* Text styling with proper contrast */
    .stMarkdown, .stText, p, div, span, label {
        color: #E5E7EB !important;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stMarkdownContainer"] {
        color: #E5E7EB !important;
    }
    
    /* Enhanced message styling */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid #22c55e !important;
        border-radius: 12px !important;
        color: #166534 !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 12px !important;
        color: #1e40af !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid #f59e0b !important;
        border-radius: 12px !important;
        color: #92400e !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid #ef4444 !important;
        border-radius: 12px !important;
        color: #dc2626 !important;
    }
    
    /* Enhanced Modern Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #818CF8 0%, #A78BFA 100%);
        color: white !important;
        border: none;
        border-radius: 16px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        letter-spacing: 0.025em;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(129, 140, 248, 0.3);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        min-height: 48px;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(129, 140, 248, 0.4);
        filter: brightness(1.15);
        background: linear-gradient(135deg, #A78BFA 0%, #818CF8 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
        box-shadow: 0 5px 15px rgba(129, 140, 248, 0.3);
    }
    
    /* Special styling for primary action buttons */
    .stButton > button:contains("Apply") {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }
    
    .stButton > button:contains("Apply"):hover {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        box-shadow: 0 15px 35px rgba(16, 185, 129, 0.4);
    }
    
    /* Special styling for find/analyze buttons */
    .stButton > button:contains("Find") {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
    }
    
    .stButton > button:contains("Find"):hover {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
        box-shadow: 0 15px 35px rgba(245, 158, 11, 0.4);
    }
    
    /* Special styling for reload/clear buttons */
    .stButton > button:contains("Reload"), .stButton > button:contains("Clear") {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
    }
    
    .stButton > button:contains("Reload"):hover, .stButton > button:contains("Clear"):hover {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        box-shadow: 0 15px 35px rgba(239, 68, 68, 0.4);
    }
    
    /* Form elements */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #0F172A !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        color: #E5E7EB !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        border-color: #818CF8 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #818CF8, #A78BFA) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 2px solid #818CF8 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5) !important;
    }
    
fig.update_traces(
    fillcolor='rgba(129, 140, 248, 0.3)',  # semi-transparent fill
    selector=dict(type='box')              # only affects box plots
)
    
    .element-container .stPlotlyChart {
        background: #0F172A !important;
    }
    fig.update_traces(
        marker=dict(size=8, opacity=0.9, line=dict(width=1, color="white"))
    )
import plotly.express as px
color_palette = px.colors.qualitative.Set2
fig = px.scatter(
    data_frame,
    x='Age',
    y='Annual Income (k$)',
    color='Cluster',
    color_discrete_sequence=color_palette,
    title='Age vs. Annual Income',
    labels={'Age': 'Age', 'Annual Income (k$)': 'Annual Income (k$)'},
    template='plotly_dark'
)

    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid #374151;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
    }
    
    .stDataFrame > div {
        background: #0F172A;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #818CF8, #A78BFA) !important;
        border-radius: 8px !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        color: #E5E7EB !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #1F2937 !important;
        border-color: #818CF8 !important;
    }
    
    .streamlit-expanderContent {
        background: #0F172A !important;
        border: 1px solid #374151 !important;
        border-top: none !important;
        color: #E5E7EB !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="metric-container"] > div {
        color: #E5E7EB !important;
    }
    
    /* Code blocks */
    .stCode {
        background: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #E5E7EB !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: #111827 !important;
        border: 2px dashed #374151 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #818CF8 !important;
        background: #1F2937 !important;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #111827;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #94A3B8;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #CBD5E1;
    }
    
    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stTabs [data-baseweb="tabpanel"] {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    if 'clustering_analyzer' not in st.session_state:
        st.session_state.clustering_analyzer = ClusteringAnalyzer()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_preprocessed' not in st.session_state:
        st.session_state.data_preprocessed = False
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = {'kmeans': False, 'dbscan': False}

def main():
    """Main application function."""
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🛍️ Customer Segmentation Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏠 Home", "📊 Data Overview", "🔍 Data Exploration", "⚙️ Preprocessing", 
        "🎯 K-Means", "🌟 DBSCAN", "📈 Comparison", "📋 Insights"
    ])
    
    # Data loading section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Data Management")
    
    # Auto-load dataset on first run
    if not st.session_state.data_loaded:
        st.session_state.data_loader.load_data()
        st.session_state.data_loaded = True
    
    # Show current dataset status
    if st.session_state.data_loaded and st.session_state.data_loader.data is not None:
        data_info = st.session_state.data_loader.get_data_info()
        st.sidebar.success(f"📊 Dataset Loaded")
        st.sidebar.info(f"**Rows:** {data_info['shape'][0]}\n**Columns:** {data_info['shape'][1]}")
        
        # Show basic info about the dataset
        if 'Annual Income (k$)' in st.session_state.data_loader.data.columns:
            st.sidebar.write("**Dataset Type:** Mall Customers")
    
    # File upload option
    st.sidebar.markdown("### 📁 Upload Different Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data_loader.data = data
            st.session_state.data_loaded = True
            st.session_state.data_preprocessed = False  # Reset preprocessing
            st.session_state.clustering_done = {'kmeans': False, 'dbscan': False}  # Reset clustering
            st.sidebar.success("✅ New file uploaded!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    
    # Reload default dataset button
    if st.sidebar.button("🔄 Reload Default Dataset"):
        st.session_state.data_loader.load_data()
        st.session_state.data_loaded = True
        st.session_state.data_preprocessed = False
        st.session_state.clustering_done = {'kmeans': False, 'dbscan': False}
        # Clear any cached clustering results
        st.session_state.clustering_analyzer = ClusteringAnalyzer()
        st.rerun()
    
    # Debug: Clear session state button (remove this after fixing)
    if st.sidebar.button("🧪 Clear Session (Debug)"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Tab content
    with tab1:
        show_home_page()
    with tab2:
        show_data_overview()
    with tab3:
        show_data_exploration()
    with tab4:
        show_preprocessing()
    with tab5:
        show_kmeans_clustering()
    with tab6:
        show_dbscan_clustering()
    with tab7:
        show_results_comparison()
    with tab8:
        show_business_insights()

def show_home_page():
    """Display the home page."""
    st.markdown('<h2 class="sub-header">Welcome to Customer Segmentation Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h3>🎯 Project Overview</h3>
        <p>This application provides a comprehensive customer segmentation analysis using machine learning clustering algorithms.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature overview
    st.markdown("### 🚀 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Data Analysis**
        - Interactive data exploration
        - Statistical summaries
        - Correlation analysis
        - Missing value detection
        """)
    
    with col2:
        st.markdown("""
        **🎯 Clustering Algorithms**
        - K-Means clustering
        - DBSCAN clustering
        - Optimal cluster determination
        - Performance metrics
        """)
    
    with col3:
        st.markdown("""
        **📈 Visualizations**
        - 2D cluster plots
        - Distribution analysis
        - Comparative visualizations
        - Interactive charts
        """)
    
    # Getting started
    st.markdown("### 🏁 Getting Started")
    st.markdown("""
    1. **📊 Data Overview**: Check your dataset information and statistics (automatically loaded from `data/Mall_Customers.csv`)
    2. **🔍 Data Exploration**: Explore distributions, correlations, and relationships
    3. **⚙️ Preprocessing**: Select features and scale your data for clustering
    4. **🎯 K-Means**: Apply K-Means clustering with optimal cluster determination
    5. **🌟 DBSCAN**: Try density-based clustering for comparison
    6. **📈 Comparison**: Compare results from both algorithms
    7. **📋 Insights**: Get business recommendations for each customer segment
    """)
    
    # Quick start note
    st.info("""
    💡 **Quick Start**: Your dataset is automatically loaded from the `data/` folder. 
    Just click on the tabs above to start exploring and clustering your customer data!
    """)
    
    # Sample data info
    st.markdown("### 📋 Sample Dataset")
    st.info("""
    The sample dataset simulates mall customer data with the following features:
    - **CustomerID**: Unique identifier
    - **Gender**: Customer gender (Male/Female)
    - **Age**: Customer age (18-70 years)
    - **Annual Income (k$)**: Annual income in thousands
    - **Spending Score (1-100)**: Mall-assigned spending score
    """)

def show_data_overview():
    """Display data overview page."""
    st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first using the sidebar.")
        return
    
    data = st.session_state.data_loader.data
    data_info = st.session_state.data_loader.get_data_info()
    
    # Basic information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", data_info['shape'][0])
    with col2:
        st.metric("Features", data_info['shape'][1])
    with col3:
        missing_values = sum(data_info['missing_values'].values())
        st.metric("Missing Values", missing_values)
    with col4:
        numeric_cols = len([col for col, dtype in data_info['dtypes'].items() if dtype in ['int64', 'float64']])
        st.metric("Numeric Features", numeric_cols)
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Data types and missing values
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Data Types")
        dtypes_df = pd.DataFrame(list(data_info['dtypes'].items()), columns=['Column', 'Data Type'])
        st.dataframe(dtypes_df, use_container_width=True)
    
    with col2:
        st.subheader("❓ Missing Values")
        missing_df = pd.DataFrame(list(data_info['missing_values'].items()), columns=['Column', 'Missing Count'])
        missing_df['Missing %'] = (missing_df['Missing Count'] / data_info['shape'][0] * 100).round(2)
        st.dataframe(missing_df, use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)

def show_data_exploration():
    """Display data exploration page."""
    st.markdown('<h2 class="sub-header">🔍 Data Exploration</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first using the sidebar.")
        return
    
    data = st.session_state.data_loader.data
    visualizer = st.session_state.visualizer
    
    # Generate exploration visualizations
    visualizer.plot_data_exploration(data)

def show_preprocessing():
    """Display preprocessing page."""
    st.markdown('<h2 class="sub-header">⚙️ Data Preprocessing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first using the sidebar.")
        return
    
    data = st.session_state.data_loader.data
    
    # Feature selection
    st.subheader("🎯 Feature Selection")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'CustomerID' in numeric_columns:
        numeric_columns.remove('CustomerID')
    
    selected_features = st.multiselect(
        "Select features for clustering:",
        numeric_columns,
        default=['Annual Income (k$)', 'Spending Score (1-100)'] if all(col in numeric_columns for col in ['Annual Income (k$)', 'Spending Score (1-100)']) else numeric_columns[:2]
    )
    
    if len(selected_features) < 2:
        st.error("⚠️ Please select at least 2 features for clustering.")
        return
    
    # Preprocessing options
    st.subheader("🔧 Preprocessing Options")
    
    col1, col2 = st.columns(2)
    with col1:
        handle_missing = st.selectbox("Handle missing values:", ["Fill with mean", "Drop rows", "No action"])
    with col2:
        scaling_method = st.selectbox("Scaling method:", ["StandardScaler", "MinMaxScaler", "No scaling"])
    
    # Apply preprocessing
    if st.button("🚀 Apply Preprocessing"):
        scaled_data = st.session_state.data_loader.preprocess_data(selected_features)
        
        if scaled_data is not None:
            st.session_state.data_preprocessed = True
            
            # Show preprocessing results
            st.success("✅ Data preprocessing completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Original Data")
                st.dataframe(data[selected_features].head(), use_container_width=True)
            
            with col2:
                st.subheader("🔄 Scaled Data")
                scaled_df = pd.DataFrame(scaled_data, columns=selected_features)
                st.dataframe(scaled_df.head(), use_container_width=True)
            
            # Feature statistics
            st.subheader("📈 Feature Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Data Statistics:**")
                st.dataframe(data[selected_features].describe(), use_container_width=True)
            
            with col2:
                st.write("**Scaled Data Statistics:**")
                st.dataframe(scaled_df.describe(), use_container_width=True)

def show_kmeans_clustering():
    """Display K-Means clustering page."""
    st.markdown('<h2 class="sub-header">🎯 K-Means Clustering</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_preprocessed:
        st.warning("⚠️ Please preprocess data first.")
        return
    
    data_loader = st.session_state.data_loader
    clustering_analyzer = st.session_state.clustering_analyzer
    visualizer = st.session_state.visualizer
    
    # Optimal cluster determination
    st.subheader("🔍 Optimal Cluster Determination")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        max_clusters = st.slider("Maximum clusters to test:", 2, 15, 10)
    
    with col2:
        if st.button("🔍 Find Optimal Clusters"):
            with st.spinner("Finding optimal number of clusters..."):
                optimization_results = clustering_analyzer.find_optimal_clusters(data_loader.scaled_data, max_clusters)
                if optimization_results:
                    visualizer.plot_optimization_results(optimization_results)
    
    # K-Means clustering
    st.subheader("🎯 K-Means Clustering")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_clusters = st.slider("Number of clusters:", 2, 10, clustering_analyzer.optimal_clusters or 5)
    
    with col2:
        if st.button("🚀 Apply K-Means"):
            # Clear any existing clustering results first to avoid column naming issues
            clustering_analyzer.cluster_labels = {}
            st.session_state.clustering_done = {'kmeans': False, 'dbscan': False}
            
            # Clear any cached data
            if hasattr(st.session_state, 'cluster_analysis_cache'):
                del st.session_state.cluster_analysis_cache
            
            with st.spinner("🔄 Applying K-Means clustering..."):
                kmeans_results = clustering_analyzer.apply_kmeans(data_loader.scaled_data, n_clusters)
            
            if kmeans_results:
                st.session_state.clustering_done['kmeans'] = True
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Silhouette Score", f"{kmeans_results['silhouette_score']:.3f}")
                with col2:
                    st.metric("Calinski-Harabasz Score", f"{kmeans_results['calinski_score']:.1f}")
                with col3:
                    st.metric("Inertia", f"{kmeans_results['inertia']:.1f}")
    
    # Visualizations
    if st.session_state.clustering_done['kmeans']:
        feature_data = data_loader.get_feature_data()
        kmeans_labels = clustering_analyzer.cluster_labels['kmeans']
        
        visualizer.plot_clusters(
            feature_data, 
            kmeans_labels, 
            'K-Means',
            data_loader.scaler,
            clustering_analyzer.kmeans_model.cluster_centers_
        )
        
        # Cluster analysis
        analysis_results = clustering_analyzer.analyze_clusters(feature_data, 'kmeans')
        if analysis_results:
            visualizer.plot_cluster_analysis(analysis_results, 'K-Means')

def show_dbscan_clustering():
    """Display DBSCAN clustering page."""
    st.markdown('<h2 class="sub-header">🌟 DBSCAN Clustering</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_preprocessed:
        st.warning("⚠️ Please preprocess data first.")
        return
    
    data_loader = st.session_state.data_loader
    clustering_analyzer = st.session_state.clustering_analyzer
    visualizer = st.session_state.visualizer
    
    # DBSCAN parameters
    st.subheader("⚙️ DBSCAN Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        eps = st.slider("Epsilon (neighborhood distance):", 0.1, 2.0, 0.5, 0.1)
    
    with col2:
        min_samples = st.slider("Minimum samples per cluster:", 2, 20, 5)
    
    # Parameter guidance
    st.info("""
    **Parameter Guidance:**
    - **Epsilon**: Maximum distance between points in the same cluster. Smaller values create more clusters.
    - **Min Samples**: Minimum number of points required to form a cluster. Higher values create fewer, denser clusters.
    """)
    
    # Apply DBSCAN
    if st.button("🚀 Apply DBSCAN"):
        dbscan_results = clustering_analyzer.apply_dbscan(data_loader.scaled_data, eps, min_samples)
        
        if dbscan_results:
            st.session_state.clustering_done['dbscan'] = True
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Clusters", dbscan_results['n_clusters'])
            with col2:
                st.metric("Noise Points", dbscan_results['n_noise'])
            with col3:
                if 'silhouette_score' in dbscan_results:
                    st.metric("Silhouette Score", f"{dbscan_results['silhouette_score']:.3f}")
                else:
                    st.metric("Silhouette Score", "N/A")
    
    # Visualizations
    if st.session_state.clustering_done['dbscan']:
        feature_data = data_loader.get_feature_data()
        dbscan_labels = clustering_analyzer.cluster_labels['dbscan']
        
        visualizer.plot_clusters(feature_data, dbscan_labels, 'DBSCAN')
        
        # Cluster analysis
        analysis_results = clustering_analyzer.analyze_clusters(feature_data, 'dbscan')
        if analysis_results:
            visualizer.plot_cluster_analysis(analysis_results, 'DBSCAN')

def show_results_comparison():
    """Display results comparison page."""
    st.markdown('<h2 class="sub-header">📈 Results Comparison</h2>', unsafe_allow_html=True)
    
    if not (st.session_state.clustering_done['kmeans'] and st.session_state.clustering_done['dbscan']):
        st.warning("⚠️ Please complete both K-Means and DBSCAN clustering first.")
        return
    
    data_loader = st.session_state.data_loader
    clustering_analyzer = st.session_state.clustering_analyzer
    visualizer = st.session_state.visualizer
    
    feature_data = data_loader.get_feature_data()
    kmeans_labels = clustering_analyzer.cluster_labels['kmeans']
    dbscan_labels = clustering_analyzer.cluster_labels['dbscan']
    
    # Comparison visualization
    visualizer.plot_comparison(feature_data, kmeans_labels, dbscan_labels)
    
    # Performance comparison
    st.subheader("📊 Performance Metrics Comparison")
    
    # Calculate metrics for both algorithms
    kmeans_analysis = clustering_analyzer.analyze_clusters(feature_data, 'kmeans')
    dbscan_analysis = clustering_analyzer.analyze_clusters(feature_data, 'dbscan')
    
    comparison_data = {
        'Metric': ['Number of Clusters', 'Silhouette Score', 'Noise Points', 'Largest Cluster Size'],
        'K-Means': [], 
        'DBSCAN': []
    }
    
    # Number of clusters
    comparison_data['K-Means'].append(len(set(kmeans_labels)))
    comparison_data['DBSCAN'].append(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))
    
    # Silhouette scores (if available)
    try:
        from sklearn.metrics import silhouette_score
        kmeans_silhouette = silhouette_score(data_loader.scaled_data, kmeans_labels)
        comparison_data['K-Means'].append(f"{kmeans_silhouette:.3f}")
        
        # DBSCAN silhouette (excluding noise)
        if -1 in dbscan_labels:
            non_noise_mask = dbscan_labels != -1
            if np.sum(non_noise_mask) > 1:
                dbscan_silhouette = silhouette_score(data_loader.scaled_data[non_noise_mask], 
                                                   dbscan_labels[non_noise_mask])
                comparison_data['DBSCAN'].append(f"{dbscan_silhouette:.3f}")
            else:
                comparison_data['DBSCAN'].append("N/A")
        else:
            dbscan_silhouette = silhouette_score(data_loader.scaled_data, dbscan_labels)
            comparison_data['DBSCAN'].append(f"{dbscan_silhouette:.3f}")
    except:
        comparison_data['K-Means'].append("N/A")
        comparison_data['DBSCAN'].append("N/A")
    
    # Noise points
    comparison_data['K-Means'].append("0")
    comparison_data['DBSCAN'].append(str(list(dbscan_labels).count(-1)))
    
    # Largest cluster size
    kmeans_counts = pd.Series(kmeans_labels).value_counts()
    dbscan_counts = pd.Series(dbscan_labels).value_counts()
    
    comparison_data['K-Means'].append(str(kmeans_counts.max()))
    if -1 in dbscan_counts.index:
        dbscan_counts = dbscan_counts.drop(-1)  # Exclude noise
    comparison_data['DBSCAN'].append(str(dbscan_counts.max()) if len(dbscan_counts) > 0 else "0")
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

def show_business_insights():
    """Display business insights page."""
    st.markdown('<h2 class="sub-header">📋 Business Insights</h2>', unsafe_allow_html=True)
    
    if not st.session_state.clustering_done['kmeans']:
        st.warning("⚠️ Please complete K-Means clustering first to generate insights.")
        return
    
    data_loader = st.session_state.data_loader
    clustering_analyzer = st.session_state.clustering_analyzer
    
    feature_data = data_loader.get_feature_data()
    
    # Generate customer profiles
    profiles = clustering_analyzer.get_cluster_profiles(feature_data, 'kmeans')
    
    if profiles:
        st.subheader("👥 Customer Segment Profiles")
        
        for profile in profiles:
            with st.expander(f"🏷️ Cluster {profile['cluster']} - {profile.get('type', 'Unknown Type')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**📊 Segment Overview**")
                    st.write(f"- **Size**: {profile['size']} customers ({profile['percentage']:.1f}%)")
                    if 'description' in profile:
                        st.write(f"- **Profile**: {profile['description']}")
                    
                    if 'avg_age' in profile:
                        st.write(f"- **Average Age**: {profile['avg_age']:.1f} ± {profile['age_std']:.1f} years")
                    
                    if 'gender_dist' in profile:
                        st.write(f"- **Gender Distribution**: {profile['gender_dist']}")
                
                with col2:
                    st.markdown(f"**💰 Financial Profile**")
                    if 'avg_income' in profile:
                        st.write(f"- **Average Income**: ${profile['avg_income']:.1f}k ± ${profile['income_std']:.1f}k")
                    
                    if 'avg_spending' in profile:
                        st.write(f"- **Average Spending Score**: {profile['avg_spending']:.1f} ± {profile['spending_std']:.1f}")
                    
                    # Business recommendations
                    st.markdown(f"**📈 Recommendations**")
                    if 'avg_income' in profile and 'avg_spending' in profile:
                        avg_income = profile['avg_income']
                        avg_spending = profile['avg_spending']
                        
                        if avg_income > 70 and avg_spending > 70:
                            st.write("- Focus on premium products and exclusive services")
                            st.write("- Implement VIP loyalty programs")
                            st.write("- Offer personalized shopping experiences")
                        elif avg_income > 70 and avg_spending < 40:
                            st.write("- Develop targeted upselling strategies")
                            st.write("- Showcase value propositions")
                            st.write("- Create incentive programs to increase spending")
                        elif avg_income < 40 and avg_spending > 70:
                            st.write("- Offer value-based products and promotions")
                            st.write("- Focus on customer retention programs")
                            st.write("- Provide flexible payment options")
                        elif avg_income < 40 and avg_spending < 40:
                            st.write("- Implement engagement and retention strategies")
                            st.write("- Offer budget-friendly options")
                            st.write("- Focus on building brand loyalty")
                        else:
                            st.write("- Balanced marketing approach")
                            st.write("- Personalized offers based on preferences")
                            st.write("- Regular engagement campaigns")
        
        # Overall business strategy
        st.subheader("🎯 Overall Business Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Marketing Strategies**
            - **Segment-specific campaigns**: Tailor marketing messages to each cluster
            - **Product positioning**: Align products with cluster preferences
            - **Channel optimization**: Use preferred communication channels per segment
            - **Pricing strategies**: Implement dynamic pricing based on segment characteristics
            """)
        
        with col2:
            st.markdown("""
            **💡 Growth Opportunities**
            - **Cross-selling**: Identify products popular in high-spending segments
            - **Retention programs**: Focus on segments with declining engagement
            - **New product development**: Create offerings for underserved segments
            - **Customer lifetime value**: Invest more in high-value segments
            """)
        
        # Download results
        st.subheader("💾 Download Results")
        
        # Prepare data for download
        result_data = feature_data.copy()
        result_data['KMeans_Cluster'] = clustering_analyzer.cluster_labels['kmeans']
        
        csv = result_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Customer Segments (CSV)",
            data=csv,
            file_name="customer_segments_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
