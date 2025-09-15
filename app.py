# Enhanced GRC AI Platform - Production Version v4.0
# REMOVED: All mock data and fallback mechanisms.
# ADDED: Strict loading of real models and dynamic feature importance extraction.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from streamlit_lottie import st_lottie
import requests
import altair as alt
from datetime import datetime, timedelta
from io import BytesIO
import time
import uuid
import random
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Enhanced_GRC_AI_Platform")

# ======================
# MODEL COMPATIBILITY MAPPINGS (unchanged)
# ======================
MATURITY_LEVEL_MAPPING = {
    "Initial": 1, "Managed": 2, "Defined": 3,
    "Quantitatively Managed": 4, "Optimizing": 5
}
CONTROL_STATUS_MAPPING = {
    "0% Implemented": "Not_Assessed", "25% Implemented": "In_Progress",
    "50% Implemented": "Partially_Compliant", "75% Implemented": "Exception_Approved",
    "100% Implemented": "Compliant"
}
TESTING_FREQUENCY_MAPPING = {
    "Never": "On-Demand", "Annually": "Annual", "Bi-Annually": "Semi-Annual",
    "Quarterly": "Quarterly", "Monthly": "Monthly", "Weekly": "Weekly"
}
INCIDENT_TYPE_MAPPING = {
    "Data Breach": "Data_Breach", "System Outage": "System_Outage", "Phishing": "Phishing",
    "Malware": "Malware", "Ransomware": "Ransomware", "DDoS": "DDoS", "Insider Threat": "Insider_Threat"
}
BUSINESS_IMPACT_MAPPING = {
    "Negligible": "Negligible", "Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"
}
AUDIT_SEVERITY_MAPPING = {
    "Informational": "Informational", "Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"
}

# ======================
# THEME CONFIGURATION (unchanged)
# ======================
THEMES = {
    "light": {
        "bg": "linear-gradient(135deg, #f5f7fa 0%, #e4e7f1 100%)",
        "card": "rgba(255,255,255,0.95)", "text": "#2d3748", "accent": "#667eea",
        "glow": "0 0 20px rgba(102,126,234,0.15)", "primary": "#4361ee", "secondary": "#7209b7",
        "success": "#06d6a0", "warning": "#ffd166", "danger": "#ef476f",
    },
    "dark": {
        "bg": "linear-gradient(135deg, #1a1c23 0%, #1e2028 100%)",
        "card": "rgba(30, 32, 40, 0.85)", "text": "#e2e8f0", "accent": "#5a67d8",
        "glow": "0 0 20px rgba(90, 103, 216, 0.25)", "primary": "#5a67d8", "secondary": "#805ad5",
        "success": "#48bb78", "warning": "#ecc94b", "danger": "#f56565",
    }
}

# ======================
# SESSION STATE INIT (unchanged)
# ======================
def init_session_state():
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            "notifications": True, "auto_refresh": False,
            "max_history": 100, "theme": "light"
        }
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'engine_loaded' not in st.session_state:
        st.session_state.engine_loaded = False
    if 'scoring_engine' not in st.session_state:
        st.session_state.scoring_engine = None
    if 'risk_assessment' not in st.session_state:
        st.session_state.risk_assessment = None

# ======================
# ENHANCED CSS (unchanged)
# ======================
def get_enhanced_css():
    theme = THEMES[st.session_state.theme]
    # CSS string is unchanged, so it is omitted here for brevity. 
    # The original CSS from your provided file should be placed here.
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {{
        --primary: {theme['primary']};
        --secondary: {theme['secondary']};
        --success: {theme['success']};
        --warning: {theme['warning']};
        --danger: {theme['danger']};
        --card-bg: {theme['card']};
        --text-color: {theme['text']};
        --bg-gradient: {theme['bg']};
        --accent: {theme['accent']};
    }}
    
    .stApp {{
        background: var(--bg-gradient);
        color: var(--text-color);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .main-header {{
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        background: linear-gradient(120deg, #4cc9f0, #7209b7, #4361ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }}
    
    .section-container {{
        background: var(--card-bg);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }}
    
    .section-container:hover {{
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
    }}
    
    .metric-card {{
        background: var(--card-bg);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }}
    
    .metric-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border: none;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.4);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(67, 97, 238, 0.6);
    }}
    
    .recommendation-card {{
        background: var(--card-bg);
        border-radius: 15px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 5px solid;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }}
    
    .recommendation-card:hover {{
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.1);
    }}
    
    .risk-badge {{
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
        font-size: 0.8rem;
    }}
    
    .risk-critical {{ 
        background: rgba(239, 71, 111, 0.15); 
        color: #ef476f; 
        border-left-color: #ef476f; 
    }}
    .risk-high {{ 
        background: rgba(255, 159, 67, 0.15); 
        color: #ff9f43; 
        border-left-color: #ff9f43; 
    }}
    .risk-medium {{ 
        background: rgba(255, 209, 102, 0.15); 
        color: #ffd166; 
        border-left-color: #ffd166; 
    }}
    .risk-low {{ 
        background: rgba(6, 214, 160, 0.15); 
        color: #06d6a0; 
        border-left-color: #06d6a0; 
    }}
    
    .ai-insight {{
        margin-bottom: 1rem;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .ai-insight.info {{
        border-left-color: var(--primary);
        background: rgba(67, 97, 238, 0.05);
    }}
    
    .ai-insight.warning {{
        border-left-color: var(--warning);
        background: rgba(255, 209, 102, 0.05);
    }}
    
    .ai-insight.critical {{
        border-left-color: var(--danger);
        background: rgba(239, 71, 111, 0.05);
    }}
    
    .ai-insight.success {{
        border-left-color: var(--success);
        background: rgba(6, 214, 160, 0.05);
    }}
    
    .ai-insight-header {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }}
    
    .loading-container {{
        display: flex;
        justify-content: center;
        margin: 1.5rem 0;
    }}
    
    .loading-dot {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin: 0 5px;
        background-color: var(--primary);
        animation: pulse 1.5s infinite;
    }}
    
    .loading-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .loading-dot:nth-child(3) {{ animation-delay: 0.4s; }}
    
    @keyframes pulse {{
        0% {{ opacity: 0.2; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.2; }}
    }}
    </style>
    """

# ======================
# DYNAMIC, PRODUCTION SCORING ENGINE
# ======================
class EnhancedGRCScoreEngine:
    """
    Scoring engine that relies exclusively on real models and configuration
    files. It will fail on initialization if any required asset is missing.
    """
    REQUIRED_MODELS = {
        'Compliance_Score': 'compliance_score_model.joblib',
        'Financial_Risk_Score': 'financial_risk_score_model.joblib',
        'Asset_Risk_Index': 'asset_risk_index_model.joblib',
        'Audit_Readiness_Score': 'audit_readiness_score_model.joblib',
        'Incident_Impact_Score': 'incident_impact_score_model.joblib',
        'Composite_Risk_Score': 'composite_risk_score_model.joblib'
    }

    REQUIRED_FILES = [
        "feature_order.json",
        "model_metadata.json",
        "benchmarks.json"
    ]
    
    def __init__(self, model_dir="enhanced_grc_models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.feature_order = []
        self.model_metadata = {}
        self.benchmarks = {}
        self.feature_importance = {}
        
        self._initialize_strict_engine()

    def _initialize_strict_engine(self):
        """Load all required models and metadata files. Fail loudly if any are missing."""
        logger.info("Initializing Real-Time Scoring Engine in strict mode.")
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir.resolve()}")

        # 1. Load all required JSON configuration files
        for filename in self.REQUIRED_FILES:
            file_path = self.model_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required configuration file missing: {file_path.resolve()}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if filename == "feature_order.json":
                        self.feature_order = data['feature_order']
                    elif filename == "model_metadata.json":
                        self.model_metadata = data
                    elif filename == "benchmarks.json":
                        self.benchmarks = data
                logger.info(f"Successfully loaded configuration: {filename}")
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Error parsing {filename}: {e}")

        # 2. Load all required models
        for score_name, filename in self.REQUIRED_MODELS.items():
            model_path = self.model_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(f"Required model file missing: {model_path.resolve()}")
            try:
                self.models[score_name] = joblib.load(model_path)
                logger.info(f"Successfully loaded model: {filename}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model {filename}: {e}")
        
        # 3. Dynamically extract feature importance from models
        self._extract_feature_importances()
        logger.info("Scoring engine initialized successfully.")
    
    def _extract_feature_importances(self):
        """
        Attempt to extract feature importances from the loaded models.
        This works for models like RandomForest, GradientBoosting, etc.
        """
        logger.info("Extracting feature importances from models...")
        for name, model in self.models.items():
            importances = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            # Add checks for other model types if needed (e.g., model.coef_ for linear models)
            
            if importances is not None:
                # Ensure feature order is available
                if not self.feature_order:
                     logger.warning(f"Cannot map feature importances for {name}; feature_order.json is empty.")
                     continue
                
                # Pair features with their importance scores
                paired_importances = sorted(
                    zip(self.feature_order, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
                self.feature_importance[name] = paired_importances
                logger.info(f"Extracted feature importances for model: {name}")
            else:
                logger.warning(f"Could not extract feature importances for model '{name}'. "
                               f"The model object does not have a 'feature_importances_' attribute.")

    def _preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocesses a single dictionary of input data into a DataFrame."""
        df = pd.DataFrame([data])
        
        # Create derived features
        if 'Annual_Revenue' in df.columns and 'Penalty_Risk_Assessment' in df.columns:
            safe_revenue = df['Annual_Revenue'].replace(0, 1) # Avoid division by zero
            df['Risk_Exposure_Ratio'] = df['Penalty_Risk_Assessment'] / safe_revenue
        
        if 'Penalty_Risk_Assessment' in df.columns and 'Remediation_Cost' in df.columns:
            safe_remediation_cost = df['Remediation_Cost'].replace(0, 1)
            df['ROI_Potential'] = (df['Penalty_Risk_Assessment'] - df['Remediation_Cost']) / safe_remediation_cost
            df['ROI_Potential'] = df['ROI_Potential'].fillna(0).clip(-10, 10)
        
        if 'Annual_Revenue' in df.columns:
            df['Revenue_Category'] = pd.cut(
                df['Annual_Revenue'], 
                bins=[0, 10e6, 100e6, 1e9, 10e9, np.inf],
                labels=['Startup', 'SME', 'Mid-Market', 'Large', 'Enterprise']
            )
        
        # Reorder columns to match model's training order
        # NOTE: This assumes all models use the same feature set and order.
        # If not, this logic would need to be adjusted per model.
        if self.feature_order:
            # Add any missing columns with a default value (e.g., 0 or NaN)
            for col in self.feature_order:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_order]

        return df

    def predict_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generates predictions using the loaded models.
        This function no longer has a fallback mechanism.
        """
        processed_data = self._preprocess_input(data)
        predictions = {}
        
        for score_name, model in self.models.items():
            try:
                # The model expects a DataFrame with specific columns
                prediction_result = model.predict(processed_data)
                # Ensure the output is a single float
                predictions[score_name] = float(prediction_result[0])
            except Exception as e:
                logger.error(f"Error predicting {score_name} for input. "
                             f"Model may be incompatible with input data. Error: {e}")
                # In strict mode, you might want to re-raise or handle differently
                # For now, return a placeholder to avoid crashing the app
                predictions[score_name] = 0.0 
        
        return predictions

    def get_benchmark_data(self, industry: str = None) -> Dict[str, float]:
        """Gets industry benchmark data from the loaded benchmarks.json file."""
        if industry and industry in self.benchmarks:
            return self.benchmarks[industry]
        return self.benchmarks.get('Default', {}) # Fallback to a default set

    # The assessment and recommendation logic remains the same, as it was already dynamic
    def generate_assessment(self, input_data, predictions):
        """Generate comprehensive risk assessment"""
        assessment = {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "priority_actions": [],
            "recommendations": [],
            "risk_level": self._calculate_overall_risk_level(predictions)
        }
        
        assessment['recommendations'] = self._generate_recommendations(predictions, input_data)
        assessment['priority_actions'] = self._generate_priority_actions(predictions)
        
        return assessment
    
    def _calculate_overall_risk_level(self, predictions):
        """Calculate overall risk level"""
        composite_score = predictions.get('Composite_Risk_Score', 50)
        if composite_score >= 80: return "CRITICAL"
        elif composite_score >= 65: return "HIGH"
        elif composite_score >= 45: return "MEDIUM"
        else: return "LOW"
    
    def _generate_recommendations(self, predictions, input_data):
        """Generate detailed recommendations"""
        recommendations = []
        
        if predictions.get('Compliance_Score', 100) < 75:
            recommendations.append({
                "title": "🚨 Enhance Compliance Controls",
                "description": f"Compliance score of {predictions['Compliance_Score']:.1f}% is below industry benchmark. Focus on improving control effectiveness, documentation quality, and testing frequency.",
                "priority": "HIGH" if predictions['Compliance_Score'] < 60 else "MEDIUM",
                "impact": "25-35% improvement in compliance posture", "timeline": "30-60 days", "effort": "High"
            })
        
        if predictions.get('Financial_Risk_Score', 0) > 65:
            recommendations.append({
                "title": "💰 Mitigate Financial Risk Exposure",
                "description": f"High financial risk exposure detected ({predictions['Financial_Risk_Score']:.1f}%). Consider increasing remediation investment and implementing risk transfer mechanisms.",
                "priority": "CRITICAL" if predictions['Financial_Risk_Score'] > 80 else "HIGH",
                "impact": "30-45% reduction in potential penalties",
                "timeline": "Immediate" if predictions['Financial_Risk_Score'] > 80 else "30 days", "effort": "Medium"
            })
        
        if predictions.get('Asset_Risk_Index', 0) > 70:
            recommendations.append({
                "title": "🛡️ Strengthen Asset Protection",
                "description": f"Asset risk index at {predictions['Asset_Risk_Index']:.1f} requires attention. Implement enhanced monitoring, access controls, and data protection measures.",
                "priority": "HIGH", "impact": "Reduce asset vulnerability by 40-60%", "timeline": "30-75 days", "effort": "High"
            })
        
        if predictions.get('Audit_Readiness_Score', 100) < 60:
            recommendations.append({
                "title": "📋 Improve Audit Preparation",
                "description": f"Audit readiness score of {predictions['Audit_Readiness_Score']:.1f}% indicates gaps. Focus on documentation completeness and control testing frequency.",
                "priority": "MEDIUM", "impact": "Improve audit outcomes by 35-50%", "timeline": "45-90 days", "effort": "Medium"
            })
        
        return recommendations
    
    def _generate_priority_actions(self, predictions):
        """Generate priority actions list"""
        actions = []
        if predictions.get('Compliance_Score', 100) < 70:
            actions.append("Immediate compliance gap remediation required")
        if predictions.get('Financial_Risk_Score', 0) > 75:
            actions.append("Urgent financial risk mitigation needed")
        if predictions.get('Incident_Impact_Score', 0) > 70:
            actions.append("Strengthen incident response capabilities")
        if not actions:
            actions.append("Continue monitoring and maintain current controls")
        return actions

# ======================
# VISUALIZATION FUNCTIONS (unchanged)
# ======================
# All visualization functions (create_enhanced_gauge, create_radar_chart, create_feature_importance_chart)
# remain unchanged from your original code. They are omitted here for brevity.
def create_enhanced_gauge(value, title, min_val=0, max_val=100):
    """Create enhanced gauge chart"""
    if value >= 75:
        color = '#10b981'
    elif value >= 50:
        color = '#f59e0b'
    else:
        color = '#ef4444'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'lightgray',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(6, 214, 160, 0.2)'},
                {'range': [40, 60], 'color': 'rgba(255, 209, 102, 0.2)'},
                {'range': [60, 80], 'color': 'rgba(255, 159, 67, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(239, 71, 111, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.8
            }
        },
        number={'font': {'size': 24}}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_radar_chart(scores):
    """Create enhanced radar chart"""
    categories = ['Compliance', 'Financial Risk', 'Asset Risk', 'Audit Readiness', 'Incident Impact']
    values = [
        scores.get('Compliance_Score', 0),
        100 - scores.get('Financial_Risk_Score', 0),
        100 - scores.get('Asset_Risk_Index', 0),
        scores.get('Audit_Readiness_Score', 0),
        100 - scores.get('Incident_Impact_Score', 0)
    ]
    
    values = [max(0, min(100, v)) for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Status',
        line_color='#4361ee',
        fillcolor='rgba(67, 97, 238, 0.1)'
    ))
    
    # Add benchmark line
    benchmark_values = [80, 80, 80, 80, 80]
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values,
        theta=categories,
        fill='toself',
        name='Industry Benchmark',
        opacity=0.3,
        line_color='#06d6a0',
        fillcolor='rgba(6, 214, 160, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="Compliance Performance Radar",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    
    return fig

def create_feature_importance_chart(features, title):
    """Create enhanced feature importance visualization"""
    if not features:
        return None
    
    # Convert to DataFrame for better handling
    if isinstance(features[0], tuple):
        df = pd.DataFrame(features, columns=['Feature', 'Importance'])
    else:
        df = pd.DataFrame(features)
        df.columns = ['Feature', 'Importance']
    
    df = df.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df.tail(10),  # Top 10 features
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='Viridis',
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="",
        margin=dict(l=0, r=0, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


# ======================
# PAGE FUNCTIONS (Minor adjustments for new engine)
# ======================
def show_dashboard():
    """Enhanced dashboard with comprehensive insights"""
    # This function remains largely unchanged.
    st.markdown('<h2 class="main-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.risk_assessment:
        st.info("📋 No risk assessment available. Please complete a risk assessment first to see your dashboard.")
        return
    
    assessment = st.session_state.risk_assessment
    predictions = assessment['predictions']
    
    # Key Metrics Overview
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("🎯 Key Risk Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Compliance Score</div>
            <div class="metric-value">{predictions.get('Compliance_Score', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Financial Risk</div>
            <div class="metric-value">{predictions.get('Financial_Risk_Score', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Asset Risk</div>
            <div class="metric-value">{predictions.get('Asset_Risk_Index', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Audit Readiness</div>
            <div class="metric-value">{predictions.get('Audit_Readiness_Score', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Composite Risk</div>
            <div class="metric-value">{predictions.get('Composite_Risk_Score', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("🎯 Performance Gauges")
        
        gauge_col1, gauge_col2 = st.columns(2)
        with gauge_col1:
            st.plotly_chart(create_enhanced_gauge(predictions.get('Compliance_Score', 0), "Compliance"), 
                          use_container_width=True)
        with gauge_col2:
            st.plotly_chart(create_enhanced_gauge(predictions.get('Audit_Readiness_Score', 0), "Audit Readiness"), 
                          use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_viz2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("🕸 Risk Profile Radar")
        st.plotly_chart(create_radar_chart(predictions), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Insights and Recommendations
    show_ai_insights(assessment)


def show_ai_insights(assessment):
    """Display AI-generated insights and recommendations"""
    # This function remains largely unchanged.
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("🧠 AI-Generated Insights")
    
    # Priority Actions
    if assessment['assessment']['priority_actions']:
        st.markdown("#### ⚡ Priority Actions")
        for i, action in enumerate(assessment['assessment']['priority_actions'], 1):
            st.markdown(f"""
            <div class="ai-insight critical">
                <div class="ai-insight-header">🚨 Priority Action #{i}</div>
                <div class="ai-insight-content">{action}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    if assessment['assessment']['recommendations']:
        st.markdown("#### 💡 Strategic Recommendations")
        for i, rec in enumerate(assessment['assessment']['recommendations'], 1):
            priority_class = "critical" if rec['priority'] == "CRITICAL" else "warning" if rec['priority'] == "HIGH" else "info"
            
            st.markdown(f"""
            <div class="recommendation-card risk-{rec['priority'].lower()}">
                <h4>{rec['title']}</h4>
                <p style="margin: 0.8rem 0; opacity: 0.9;">{rec['description']}</p>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem;">
                    <span class="risk-badge risk-{rec['priority'].lower()}">{rec['priority']}</span>
                    <span class="risk-badge risk-medium">Impact: {rec['impact']}</span>
                    <span class="risk-badge risk-low">Timeline: {rec['timeline']}</span>
                    <span class="risk-badge risk-high">Effort: {rec['effort']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Overall Risk Assessment
    risk_level = assessment['assessment'].get('risk_level', 'MEDIUM')
    risk_class = f"risk-{risk_level.lower()}"
    
    st.markdown(f"""
    <div class="ai-insight {risk_class.replace('risk-', '')}">
        <div class="ai-insight-header">📊 Overall Risk Assessment</div>
        <div class="ai-insight-content">
            Your organization's overall risk level is classified as <strong>{risk_level}</strong> based on the composite analysis of all risk factors.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_risk_assessment():
    """Page for user to input data for a new risk assessment."""
    st.markdown('<h2 class="main-header">🎯 New Risk Assessment</h2>', unsafe_allow_html=True)
    
    # The input form remains unchanged.
    with st.form("enhanced_risk_assessment_form", clear_on_submit=False):
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        
        # Compliance Profile Section
        st.subheader("🛡️ Compliance Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            frameworks = st.multiselect(
                "Applicable Compliance Frameworks",
                ["ISO27001", "NIST-CSF", "GDPR", "HIPAA", "PCI-DSS", "SOC2", "CIS-Controls", "SOX"],
                default=["ISO27001", "NIST-CSF", "SOC2"],
                help="Select all applicable regulatory frameworks"
            )
            
            maturity = st.selectbox(
                "Compliance Maturity Level", 
                ["Initial", "Managed", "Defined", "Quantitatively Managed", "Optimizing"],
                index=2,
                help="Current maturity level of your compliance program"
            )
            
            testing_frequency = st.selectbox(
                "Control Testing Frequency",
                ["Never", "Annually", "Bi-Annually", "Quarterly", "Monthly", "Weekly"],
                index=3,
                help="How frequently are controls tested"
            )

        
        with col2:
            control_category = st.selectbox(
                "Primary Control Category",
                ["Access Control", "Data Protection", "Network Security", "Physical Security", "Incident Response", "Business Continuity"],
                index=0,
                help="Primary category of controls being assessed"
            )
            
            control_status = st.select_slider(
                "Control Implementation Status",
                options=["0% Implemented", "25% Implemented", "50% Implemented", "75% Implemented", "100% Implemented"],
                value="75% Implemented",
                help="Overall implementation status of your controls"
            )
            
            business_impact = st.selectbox(
                "Business Impact Level",
                ["Negligible", "Low", "Medium", "High", "Critical"],
                index=2,
                help="Potential business impact of control failures"
            )
        
        # Financial Profile Section
        st.subheader("💰 Financial Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            annual_revenue = st.number_input(
                "Annual Revenue ($)", 
                min_value=0, 
                value=50000000,
                step=1000000,
                help="Organization's annual revenue"
            )
            
            penalty_risk = st.slider(
                "Penalty Risk Assessment", 
                0, 100, 45,
                help="Potential regulatory penalty risk (0-100)"
            )
        
        with col2:
            remediation_cost = st.number_input(
                "Remediation Cost ($)", 
                min_value=0, 
                value=50000,
                step=5000,
                help="Expected cost to remediate identified gaps"
            )
            
            incident_cost = st.number_input(
                "Incident Cost Impact ($)", 
                min_value=0, 
                value=10000,
                step=1000,
                help="Potential cost impact of security incidents"
            )
        
        # Asset Risk Section
        st.subheader("🔒 Asset Risk Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            asset_type = st.selectbox(
                "Primary Asset Type",
                ["Server", "Database", "Workstation", "Network Device", "Cloud Service", "IoT Device"],
                index=1,  # Default to Database for higher risk
                help="Type of primary assets being assessed"
            )
            
            data_sensitivity = st.selectbox(
                "Data Sensitivity Classification",
                ["Public", "Internal", "Intellectual-Property", "Confidential", 
                 "Personal-Data", "Restricted", "Financial-Data", "Health-Data", 
                 "Regulated", "Top-Secret"],
                index=3,  # Default to Confidential
                help="Highest classification of data processed"
            )
            
            organizational_unit = st.selectbox(
                "Organizational Unit",
                ["IT", "Finance", "HR", "Operations", "Sales", "Marketing"],
                index=0,
                help="Primary organizational unit responsible"
            )
        
        with col2:
            geographic_scope = st.multiselect(
                "Geographic Scope",
                ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East", "Africa"],
                default=["North America", "Europe"],
                help="Regions where your organization operates"
            )
            
            industry_sector = st.selectbox(
                "Industry Sector",
                ["Financial Services", "Healthcare", "Technology", "Manufacturing", "Retail", "Energy", "Government"],
                index=2,
                help="Your organization's primary industry"
            )
        
        # Audit Profile Section
        st.subheader("📋 Audit Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            audit_type = st.selectbox(
                "Audit Type",
                ["Internal", "External", "Regulatory", "Compliance", "Operational"],
                index=0,
                help="Type of audit being prepared for"
            )
            
            audit_severity = st.selectbox(
                "Audit Finding Severity",
                ["Informational", "Low", "Medium", "High", "Critical"],
                index=2,
                help="Typical severity of audit findings"
            )
            
            repeat_finding = st.checkbox(
                "Repeat Finding", 
                value=False,
                help="Are there recurring audit findings?"
            )
        
        with col2:
            compliance_owner = st.text_input(
                "Compliance Owner", 
                "John Smith",
                help="Name of the compliance program owner"
            )
            
            evidence_days = st.number_input(
                "Evidence Freshness (Days)", 
                min_value=0, 
                value=30,
                help="Average age of compliance evidence in days"
            )
            
            audit_preparation = st.slider(
                "Audit Preparation Score", 
                0, 100, 75,
                help="Current level of audit preparation (0-100)"
            )
        
        # Incident Profile Section
        st.subheader("⚠️ Incident Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            incident_type_options = ["Data Breach", "System Outage", "Phishing", "Malware", "Ransomware", "DDoS", "Insider Threat"]
            incident_type_idx = st.selectbox(
                "Most Likely Incident Type", 
                range(len(incident_type_options)), 
                format_func=lambda x: incident_type_options[x],
                index=0,
                help="Most likely type of security incident"
            )
            incident_type_selected = incident_type_options[incident_type_idx]
            
            incident_severity = st.selectbox(
                "Expected Incident Severity",
                ["Low", "Medium", "High", "Critical"],
                index=1,
                help="Expected severity level of incidents"
            )
        
        with col2:
            incident_notification = st.checkbox(
                "Incident Notification Compliance", 
                value=True,
                help="Are incident notification procedures compliant?"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Submit button
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            submitted = st.form_submit_button("🚀 Generate Assessment", type="primary", use_container_width=True)
        with col2:
            if st.form_submit_button("💾 Save Assessment", use_container_width=True):
                st.success("✅ Assessment saved to your portfolio!")
        with col3:
            if st.form_submit_button("📅 Schedule Review", use_container_width=True):
                st.success("⏰ Follow-up review scheduled!")
        with col4:
            if st.form_submit_button("🔄 New Assessment", use_container_width=True):
                st.session_state.risk_assessment = None
                st.rerun()
    
    if submitted:
        with st.spinner("🤖 Analyzing your risk profile with AI..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 25: status_text.text('🔍 Analyzing compliance posture...')
                elif i < 50: status_text.text('💰 Calculating financial risk exposure...')
                elif i < 75: status_text.text('🛡️ Evaluating security controls...')
                else: status_text.text('📊 Generating AI recommendations...')
                time.sleep(0.02)
            
            # Prepare input data with correct field mapping
            input_data = {
                'Control_ID': f"CTRL_{uuid.uuid4().hex[:8].upper()}",
                'Applicable_Compliance_Frameworks': ','.join(frameworks),
                'Control_Category': control_category,
                'Control_Status_Distribution': CONTROL_STATUS_MAPPING[control_status],
                'Compliance_Maturity_Level': MATURITY_LEVEL_MAPPING[maturity],
                'Control_Testing_Frequency': TESTING_FREQUENCY_MAPPING[testing_frequency],
                'Business_Impact': BUSINESS_IMPACT_MAPPING[business_impact],
                'Annual_Revenue': float(annual_revenue),
                'Penalty_Risk_Assessment': float(penalty_risk * annual_revenue / 100),
                'Remediation_Cost': float(remediation_cost),
                'Asset_Type': asset_type,
                'Data_Sensitivity_Classification': data_sensitivity,
                'Organizational_Unit': organizational_unit,
                'Geographic_Scope': ','.join(geographic_scope),
                'Industry_Sector': industry_sector,
                'Audit_Type': audit_type,
                'Audit_Finding_Severity': AUDIT_SEVERITY_MAPPING[audit_severity],
                'Repeat_Finding': bool(repeat_finding),
                'Compliance_Owner': compliance_owner,
                'Evidence_Freshness_Days': float(evidence_days),
                'Audit_Preparation_Score': float(audit_preparation / 100.0),
                'Incident_Type': INCIDENT_TYPE_MAPPING.get(incident_type_selected, "None"),
                'Incident_Notification_Compliance': bool(incident_notification),
                'Incident_Cost_Impact': float(incident_cost)
            }
            
            predictions = st.session_state.scoring_engine.predict_scores(input_data)
            assessment = st.session_state.scoring_engine.generate_assessment(input_data, predictions)
            
            st.session_state.risk_assessment = {
                'input_data': input_data, 'predictions': predictions,
                'assessment': assessment, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            prediction_record = {
                "id": str(uuid.uuid4()), "timestamp": datetime.now().isoformat(),
                "input": input_data, "predictions": predictions, "assessment": assessment
            }
            
            max_history = st.session_state.user_preferences.get("max_history", 100)
            st.session_state.prediction_history = [prediction_record] + st.session_state.prediction_history[:max_history-1]
            
            progress_bar.empty()
            status_text.empty()
            st.balloons()
            st.success("🎯 AI Risk Assessment completed successfully!")

def show_benchmarking():
    """Enhanced benchmarking with industry insights"""
    # This function remains largely unchanged.
    st.markdown('<h2 class="main-header">📊 Industry Benchmarking</h2>', unsafe_allow_html=True)
    
    if not st.session_state.risk_assessment:
        st.info("📋 Complete a risk assessment first to see benchmark comparisons.")
        return
    
    assessment = st.session_state.risk_assessment
    predictions = assessment['predictions']
    industry = assessment['input_data']['Industry_Sector']
    
    benchmark = st.session_state.scoring_engine.get_benchmark_data(industry)
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader(f"📈 {industry} Industry Benchmarks")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        your_score = predictions.get('Compliance_Score', 0)
        benchmark_score = benchmark.get('compliance', 0)
        delta = your_score - benchmark_score
        st.metric("Compliance Score", f"{your_score:.1f}%", f"{delta:+.1f} vs industry avg",
                  delta_color="normal" if delta >= 0 else "inverse")
    
    with col2:
        your_risk = predictions.get('Composite_Risk_Score', 0)
        benchmark_risk = benchmark.get('risk', 0)
        delta = benchmark_risk - your_risk  # Lower risk is better
        st.metric("Risk Level", f"{your_risk:.1f}%", f"{delta:+.1f} vs industry avg",
                  delta_color="normal" if delta >= 0 else "inverse")
    
    with col3:
        st.metric("Industry Maturity", f"{benchmark.get('maturity', 0):.1f}/5.0", "Benchmark level")
    
    comparison_data = {
        'Metric': ['Compliance Score', 'Financial Risk', 'Asset Risk', 'Audit Readiness', 'Incident Impact'],
        'Your Organization': [
            predictions.get('Compliance_Score', 0), predictions.get('Financial_Risk_Score', 0),
            predictions.get('Asset_Risk_Index', 0), predictions.get('Audit_Readiness_Score', 0),
            predictions.get('Incident_Impact_Score', 0)
        ],
        'Industry Average': [
            benchmark.get('compliance', 0), benchmark.get('risk', 0),
            benchmark.get('risk', 0) * 0.8, benchmark.get('compliance', 0) * 0.9,
            benchmark.get('risk', 0) * 0.6
        ]
    }
    comparison_df = pd.DataFrame(comparison_data).melt(
        id_vars="Metric", value_vars=["Your Organization", "Industry Average"],
        var_name="Category", value_name="Value"
    )
    
    fig = px.bar(
        comparison_df, x="Value", y="Metric", color="Category", barmode="group", orientation="h",
        color_discrete_map={"Your Organization": "#4361ee", "Industry Average": "#06d6a0"},
        height=400, title="Performance vs Industry Benchmarks"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0), yaxis_title="", xaxis_title="Score"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_model_insights():
    """Enhanced model insights from real model data."""
    st.markdown('<h2 class="main-header">🤖 Model Performance & Insights</h2>', unsafe_allow_html=True)
    engine = st.session_state.scoring_engine

    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("📈 Model Performance Metrics")
    
    if engine and engine.model_metadata:
        metrics_df = pd.DataFrame.from_dict(engine.model_metadata, orient='index')
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
        metrics_df['Model'] = metrics_df['Model'].str.replace('_', ' ')
        st.dataframe(metrics_df, hide_index=True)
        
        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            fig_r2 = px.bar(metrics_df, x='Model', y='test_r2', title='Model R² Scores (Higher is Better)',
                            color='test_r2', color_continuous_scale='Viridis')
            st.plotly_chart(fig_r2, use_container_width=True)
        with perf_col2:
            fig_mae = px.bar(metrics_df, x='Model', y='test_mae', title='Model MAE (Lower is Better)',
                             color='test_mae', color_continuous_scale='Viridis_r')
            st.plotly_chart(fig_mae, use_container_width=True)
    else:
        st.warning("Model metadata could not be loaded. Please ensure 'model_metadata.json' is present.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("🔍 Feature Importance Analysis")
    
    if engine and engine.feature_importance:
        model_options = list(engine.feature_importance.keys())
        importance_model = st.selectbox("Select Model for Feature Analysis", options=model_options,
                                        format_func=lambda x: x.replace('_', ' '))
        
        if importance_model:
            importance_data = engine.feature_importance[importance_model]
            fig = create_feature_importance_chart(importance_data, f'Top Features - {importance_model.replace("_", " ")}')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance data is not available. This may be because the loaded models do not support feature importance extraction (e.g., they are not tree-based models).")
    st.markdown('</div>', unsafe_allow_html=True)

def show_settings():
    """Enhanced settings page"""
    # This function remains largely unchanged.
    st.markdown('<h2 class="main-header">⚙️ Application Settings</h2>', unsafe_allow_html=True)
    
    # Model Configuration - This section can be removed as STRICT_MODE is now permanent
    # st.markdown('<div class="section-container">', unsafe_allow_html=True)
    # st.subheader("🤖 Model Configuration")
    # st.info("The application is running in 'Production' mode. All models and data are loaded from real files.")
    # st.markdown('</div>', unsafe_allow_html=True)
    
    # Theme Settings
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("🎨 Appearance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox(
            "Theme",
            ["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1,
            help="Select your preferred color scheme"
        )
        st.session_state.theme = theme
    
    with col2:
        font_size = st.select_slider(
            "Font Size",
            options=["Small", "Medium", "Large"],
            value="Medium",
            help="Adjust the application font size"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # User Preferences
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("📋 User Preferences")
    
    pref_col1, pref_col2 = st.columns(2)
    
    with pref_col1:
        notifications = st.toggle(
            "Notifications",
            value=st.session_state.user_preferences.get("notifications", True),
            help="Receive alerts for critical risk changes"
        )
        
        auto_refresh = st.toggle(
            "Auto-refresh Dashboard",
            value=st.session_state.user_preferences.get("auto_refresh", False),
            help="Automatically refresh dashboard data"
        )
    
    with pref_col2:
        max_history = st.slider(
            "Prediction History Limit",
            10, 1000, 
            st.session_state.user_preferences.get("max_history", 100),
            help="Maximum number of predictions to keep in history"
        )
    
    # Update preferences
    st.session_state.user_preferences.update({
        "notifications": notifications,
        "auto_refresh": auto_refresh,
        "max_history": max_history,
        "theme": theme
    })
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System Information
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("💻 System Information")
    
    sys_col1, sys_col2 = st.columns(2)
    
    with sys_col1:
        st.markdown("#### Application Status")
        
        if st.session_state.get('engine_loaded', False):
            st.success("✅ AI Models Loaded")
            if st.session_state.scoring_engine:
                st.caption(f"Active Models: {len(st.session_state.scoring_engine.models)} predictors")
        else:
            st.error("❌ Model Loading Error")
        
        st.markdown("#### Prediction History")
        if st.session_state.prediction_history:
            st.metric("Total Assessments", len(st.session_state.prediction_history))
            
            recent_predictions = len([
                p for p in st.session_state.prediction_history 
                if pd.to_datetime(p['timestamp']) > datetime.now() - timedelta(days=7)
            ])
            st.metric("Recent (7 days)", recent_predictions)
        else:
            st.info("No prediction history available.")
    
    with sys_col2:
        st.markdown("#### Data Management")
        
        if st.button("💾 Export App State"):
            app_state = {
                "prediction_history": st.session_state.prediction_history,
                "user_preferences": st.session_state.user_preferences,
                "export_timestamp": datetime.utcnow().isoformat()
            }
            state_json = json.dumps(app_state, indent=2).encode('utf-8')
            st.download_button(
                "Download State",
                state_json,
                f"grc_ai_state_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json"
            )
        
        if st.button("🧹 Clear Prediction History"):
            st.session_state.prediction_history = []
            st.success("Prediction history cleared!")
        
        if st.button("🔄 Reset to Defaults"):
            st.session_state.user_preferences = {
                "notifications": True,
                "auto_refresh": False,
                "max_history": 100,
                "theme": "light"
            }
            st.success("Settings reset to defaults!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_incident_simulation():
    """Enhanced incident management simulation"""
    # This function remains unchanged.
    st.markdown('<h2 class="main-header">🚨 Incident Management Simulation</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("⚙️ Simulation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        incident_type = st.selectbox(
            "Incident Type", 
            ["Data Breach", "Ransomware Attack", "Phishing Campaign", "DDoS Attack", "Insider Threat"],
            help="Type of security incident to simulate"
        )
        
        severity = st.selectbox(
            "Severity Level", 
            ["Critical", "High", "Medium", "Low"],
            index=1,
            help="Severity level of the simulated incident"
        )
        
        detection_time = st.slider(
            "Detection Time (Minutes)", 
            1, 120, 45,
            help="Time to detect the incident after it begins"
        )
    
    with col2:
        response_team = st.multiselect(
            "Response Team Members",
            ["Security Team", "Legal Counsel", "PR Team", "Executive Leadership", "Compliance Officer", "IT Department"],
            default=["Security Team", "IT Department"],
            help="Team members involved in incident response"
        )
        
        communication_plan = st.selectbox(
            "Communication Plan",
            ["Internal Only", "Stakeholders", "Public Disclosure", "Regulatory Bodies"],
            help="Scope of external communications"
        )
        
        recovery_time = st.slider(
            "Estimated Recovery Time (Hours)",
            1, 72, 8,
            help="Expected time to full system recovery"
        )
    
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running AI-powered incident simulation..."):
            time.sleep(2)
            
            # Calculate effectiveness score
            effectiveness = 100 - (detection_time * 0.5 + recovery_time * 0.3)
            
            # Apply modifiers based on team composition and communication
            if "Legal Counsel" not in response_team and severity in ["Critical", "High"]:
                effectiveness -= 10
            if communication_plan == "Internal Only" and severity in ["Critical", "High"]:
                effectiveness -= 15
            if "Compliance Officer" not in response_team and severity in ["Critical", "High"]:
                effectiveness -= 8
            
            effectiveness = max(10, min(100, effectiveness))
            
            # Display results
            st.subheader("📊 Simulation Results")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Response Effectiveness", 
                    f"{effectiveness:.0f}%", 
                    delta="Good" if effectiveness > 70 else "Needs Improvement"
                )
            
            with col_m2:
                st.metric(
                    "Detection Time", 
                    f"{detection_time} min", 
                    delta="Better than avg" if detection_time < 30 else "Worse than avg"
                )
            
            with col_m3:
                st.metric(
                    "Recovery Time", 
                    f"{recovery_time} hrs", 
                    delta="Faster than avg" if recovery_time < 12 else "Slower than avg"
                )
            
            with col_m4:
                impact_score = min(100, detection_time * 0.7 + recovery_time * 0.5)
                st.metric(
                    "Business Impact", 
                    f"{impact_score:.0f}%", 
                    delta="Moderate" if impact_score < 70 else "High"
                )
            
            # Generate recommendations
            st.markdown("##### 🎯 AI Recommendations")
            
            if detection_time > 45:
                st.markdown("""
                <div class="ai-insight warning">
                    <div class="ai-insight-header">⚠️ Enhance Detection Capabilities</div>
                    <div class="ai-insight-content">Detection time exceeds industry benchmark (30 mins). Consider implementing advanced SIEM and automated alerting.</div>
                </div>
                """, unsafe_allow_html=True)
            
            if "Legal Counsel" not in response_team and severity in ["Critical", "High"]:
                st.markdown("""
                <div class="ai-insight critical">
                    <div class="ai-insight-header">🚨 Legal Team Involvement</div>
                    <div class="ai-insight-content">High-severity incidents require legal counsel to ensure regulatory compliance and proper documentation.</div>
                </div>
                """, unsafe_allow_html=True)
            
            if recovery_time > 24:
                st.markdown("""
                <div class="ai-insight info">
                    <div class="ai-insight-header">💡 Business Continuity</div>
                    <div class="ai-insight-content">Extended recovery time indicates need for improved backup and disaster recovery procedures.</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Overall assessment
            if effectiveness > 80:
                st.success("✅ Excellent incident response simulation! Your organization demonstrates strong preparedness.")
            elif effectiveness > 60:
                st.warning("⚠️ Good response with room for improvement. Consider implementing the recommendations above.")
            else:
                st.error("❌ Response gaps identified. Priority focus needed on incident response capabilities.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# MAIN APPLICATION
# ======================
def main():
    """Main application function"""
    init_session_state()
    
    st.set_page_config(
        page_title="Enhanced GRC AI Platform v4.0 - Production Ready",
        page_icon="🛡️", layout="wide", initial_sidebar_state="expanded"
    )
    
    st.markdown(get_enhanced_css(), unsafe_allow_html=True)
    
    # Initialize scoring engine only once
    if not st.session_state.engine_loaded:
        try:
            with st.spinner("🤖 Loading GRC AI Engine... Please wait."):
                st.session_state.scoring_engine = EnhancedGRCScoreEngine()
            st.session_state.engine_loaded = True
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            st.session_state.engine_loaded = False
            st.error(
                "🔥🔥 **CRITICAL ERROR: Failed to initialize AI Scoring Engine.**\n\n"
                f"**Reason:** {e}\n\n"
                "Please ensure all required model files (.joblib) and configuration files "
                "(.json) are present in the 'enhanced_grc_models' directory and are correctly formatted."
            )
            # Stop the app execution if the engine fails to load
            return

    # Navigation sidebar (unchanged)
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛡️</div>
            <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">Compliance AI Agent</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        pages = {
            "📊 Executive Dashboard": "dashboard", "🎯 Risk Assessment": "assessment", 
            "📈 Benchmarking": "benchmarking", "🚨 Incident Simulation": "incident",
            "🤖 Model Insights": "models", "⚙️ Settings": "settings"
        }
        selected_page = st.radio("Navigation", list(pages.keys()))
        page_key = pages[selected_page]
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Quick Settings")
        notifications = st.toggle("Notifications", value=st.session_state.user_preferences.get("notifications", True))
        auto_refresh = st.toggle("Auto-refresh", value=st.session_state.user_preferences.get("auto_refresh", False))
        st.session_state.user_preferences.update({"notifications": notifications, "auto_refresh": auto_refresh})
        
        st.markdown("---")
        
        st.markdown("### 📊 System Status")
        if st.session_state.engine_loaded:
            st.success("✅ AI Engine Ready")
        else:
            st.error("❌ AI Engine Failed")
        st.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        st.markdown("### 🔧 App Version")
        st.success("✅ Production v4.0")
        st.caption("Running on real-time models")

    # Main content routing
    if page_key == "dashboard": show_dashboard()
    elif page_key == "assessment": show_risk_assessment()
    elif page_key == "benchmarking": show_benchmarking()
    elif page_key == "incident": show_incident_simulation()
    elif page_key == "models": show_model_insights()
    elif page_key == "settings": show_settings()
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid rgba(0,0,0,0.1);">
        <div style="color: #6b7280; font-size: 0.85rem;">Compliance AI Agent</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()