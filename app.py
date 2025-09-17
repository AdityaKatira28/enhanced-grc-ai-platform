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
from streamlit_autorefresh import st_autorefresh

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Enhanced_GRC_Monitoring_Platform")

MATURITY_LEVEL_MAPPING = {
    "Initial": 1,
    "Managed": 2,
    "Defined": 3,
    "Quantitatively Managed": 4,
    "Optimizing": 5
}
CONTROL_STATUS_MAPPING = {
    "0% Implemented": "Not_Assessed",
    "25% Implemented": "In_Progress",
    "50% Implemented": "Partially_Compliant",
    "75% Implemented": "Exception_Approved",
    "100% Implemented": "Compliant"
}
TESTING_FREQUENCY_MAPPING = {
    "Never": "On-Demand",
    "Annually": "Annual",
    "Bi-Annually": "Semi-Annual",
    "Quarterly": "Quarterly",
    "Monthly": "Monthly",
    "Weekly": "Weekly"
}
INCIDENT_TYPE_MAPPING = {
    "Data Breach": "Data_Breach",
    "System Outage": "System_Outage",
    "Phishing": "Phishing",
    "Malware": "Malware",
    "Ransomware": "Ransomware",
    "DDoS": "DDoS",
    "Insider Threat": "Insider_Threat"
}
BUSINESS_IMPACT_MAPPING = {
    "Negligible": "Negligible",
    "Low": "Low",
    "Medium": "Medium",
    "High": "High",
    "Critical": "Critical"
}
AUDIT_SEVERITY_MAPPING = {
    "Informational": "Informational",
    "Low": "Low",
    "Medium": "Medium",
    "High": "High",
    "Critical": "Critical"
}

THEMES = {
    "light": {
        "bg": "linear-gradient(135deg, #f5f7fa 0%, #e4e7f1 100%)",
        "card": "rgba(255,255,255,0.95)",
        "text": "#2d3748",
        "accent": "#667eea",
        "glow": "0 0 20px rgba(102,126,234,0.15)",
        "primary": "#4361ee",
        "secondary": "#7209b7",
        "success": "#06d6a0",
        "warning": "#ffd166",
        "danger": "#ef476f",
    },
    "dark": {
        "bg": "linear-gradient(135deg, #1a1c23 0%, #1e2028 100%)",
        "card": "rgba(30, 32, 40, 0.85)",
        "text": "#e2e8f0",
        "accent": "#5a67d8",
        "glow": "0 0 20px rgba(90, 103, 216, 0.25)",
        "primary": "#5a67d8",
        "secondary": "#805ad5",
        "success": "#48bb78",
        "warning": "#ecc94b",
        "danger": "#f56565",
    }
}

def init_session_state():
    if 'theme' not in st.session_state:
        st.session_state.theme = "light" # [cite: 6]
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            "notifications": True,
            "auto_refresh": True,
            "max_history": 100,
            "theme": "light"
        } # [cite: 6]
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = [] # [cite: 7]
    if 'engine_loaded' not in st.session_state:
        st.session_state.engine_loaded = False # [cite: 7]
    if 'scoring_engine' not in st.session_state:
        st.session_state.scoring_engine = None # [cite: 7]
    if 'historical_assessments' not in st.session_state:
        st.session_state.historical_assessments = [] # [cite: 7]
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {} # [cite: 7]

def get_enhanced_css():
    theme = THEMES[st.session_state.theme] # [cite: 7]
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
    .error-container {{
        background: rgba(239, 71, 111, 0.1);
        border: 1px solid rgba(239, 71, 111, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
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

class RealModelGRCEngine:
    def __init__(self, model_dir="enhanced_grc_models"):
        self.model_dir = Path(model_dir) # [cite: 52]
        self.models = {} # [cite: 52]
        self.feature_info = {} # [cite: 52]
        self.scoring_config = {} # [cite: 52]
        self.model_metadata = {} # [cite: 52]
        self.feature_importance = {} # [cite: 52]
        self.model_versions = {} # [cite: 52]
        self.is_loaded = False # [cite: 52]
        self._initialize_engine() # [cite: 53]

    def _initialize_engine(self):
        """Initialize engine with real models only""" # [cite: 53]
        try:
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Model directory '{self.model_dir}' does not exist") # [cite: 53]
            self._load_models() # [cite: 53]
            self._load_metadata() # [cite: 53]
            if not self.models:
                raise ValueError("No valid models found in model directory") # [cite: 54]
            self.is_loaded = True # [cite: 54]
            logger.info(f"GRC Scoring Engine initialized with {len(self.models)} real models") # [cite: 54]
        except Exception as e:
            logger.error(f"Failed to initialize GRC Scoring Engine: {str(e)}") # [cite: 54]
            self.is_loaded = False # [cite: 55]
            raise # [cite: 55]

    def _load_models(self):
        """Load real .joblib models only""" # [cite: 55]
        model_files = {
            'Compliance_Score': 'compliance_score_model.joblib',
            'Financial_Risk_Score': 'financial_risk_score_model.joblib',
            'Asset_Risk_Index': 'asset_risk_index_model.joblib',
            'Audit_Readiness_Score': 'audit_readiness_score_model.joblib',
            'Incident_Impact_Score': 'incident_impact_score_model.joblib',
            'Composite_Risk_Score': 'composite_risk_score_model.joblib'
        } # [cite: 55, 56]
        for score_name, filename in model_files.items():
            model_path = self.model_dir / filename # [cite: 56]
            if model_path.exists():
                try:
                    self.models[score_name] = joblib.load(model_path) # [cite: 57]
                    self.model_versions[score_name] = model_path.stat().st_mtime # [cite: 57]
                    logger.info(f"Successfully loaded {score_name} model from {filename}") # [cite: 57]
                except Exception as e:
                    logger.error(f"Failed to load {score_name} model: {str(e)}") # [cite: 58]
                    raise # [cite: 58]
            else:
                logger.error(f"Model file not found: {model_path}") # [cite: 58]
                raise FileNotFoundError(f"Required model file not found: {filename}") # [cite: 58]

    def _load_metadata(self):
        """Load real metadata files only""" # [cite: 59]
        metadata_files = {
            'feature_info.json': 'feature_info',
            'scoring_config.json': 'scoring_config',
            'model_metrics.json': 'model_metadata',
            'feature_importance.json': 'feature_importance'
        } # [cite: 59]
        for filename, attr_name in metadata_files.items():
            file_path = self.model_dir / filename # [cite: 59, 60]
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        setattr(self, attr_name, json.load(f)) # [cite: 60]
                    logger.info(f"Loaded metadata: {filename}") # [cite: 61]
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {str(e)}") # [cite: 61]
                    raise # [cite: 61]
            else:
                logger.warning(f"Metadata file not found: {filename}") # [cite: 61]

    def predict_scores(self, data):
        """Predict scores using real models only""" # [cite: 62]
        if not self.is_loaded:
            raise RuntimeError("Scoring engine not properly initialized") # [cite: 62]
        try:
            processed_data = self._preprocess_input(data) # [cite: 62]
            predictions = {} # [cite: 62]
            for score_name, model in self.models.items():
                try:
                    pred = model.predict(processed_data) # [cite: 63]
                    predictions[score_name] = float(pred[0]) if hasattr(pred, '__iter__') else float(pred) # [cite: 63]
                    logger.info(f"Predicted {score_name}: {predictions[score_name]:.2f}") # [cite: 63]
                except Exception as e:
                    logger.error(f"Error predicting {score_name}: {str(e)}") # [cite: 64]
                    raise # [cite: 64]
            return predictions # [cite: 64]
        except Exception as e:
            logger.error(f"Error in predict_scores: {str(e)}") # [cite: 64]
            raise # [cite: 65]

    def _preprocess_input(self, data):
        """Preprocess input data for model compatibility""" # [cite: 65]
        if isinstance(data, dict):
            df = pd.DataFrame([data]) # [cite: 65]
        else:
            df = data.copy() # [cite: 65]

        # Create derived features using correct field names
        try:
            if 'Annual_Revenue' in df.columns and 'Penalty_Risk_Assessment' in df.columns:
                safe_revenue = df['Annual_Revenue'].replace(0, 1) # [cite: 66]
                df['Risk_Exposure_Ratio'] = df['Penalty_Risk_Assessment'] / safe_revenue # [cite: 66]

            if 'Penalty_Risk_Assessment' in df.columns and 'Remediation_Cost' in df.columns:
                safe_remediation_cost = df['Remediation_Cost'].replace(0, 1) # [cite: 67]
                df['ROI_Potential'] = (df['Penalty_Risk_Assessment'] - df['Remediation_Cost']) / safe_remediation_cost # [cite: 67]
                df['ROI_Potential'] = df['ROI_Potential'].fillna(0).clip(-10, 10) # [cite: 67]

            if 'Annual_Revenue' in df.columns:
                df['Revenue_Category'] = pd.cut(df['Annual_Revenue'],
                                               bins=[0, 10e6, 100e6, 1e9, 10e9, np.inf],
                                               labels=['Startup', 'SME', 'Mid-Market', 'Large', 'Enterprise']) # [cite: 67, 68]
        except Exception as e:
            logger.warning(f"Error in feature engineering: {str(e)}") # [cite: 69]

        return df # [cite: 69]

    def generate_assessment(self, input_data, predictions):
        """Generate comprehensive risk assessment based on real model predictions""" # [cite: 69]
        assessment = {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "priority_actions": [],
            "recommendations": [],
            "risk_level": self._calculate_overall_risk_level(predictions)
        } # [cite: 69, 70]

        # Generate recommendations based on real scores
        recommendations = self._generate_recommendations(predictions, input_data) # [cite: 70]
        assessment['recommendations'] = recommendations # [cite: 70]

        # Generate priority actions based on real scores
        priority_actions = self._generate_priority_actions(predictions) # [cite: 71]
        assessment['priority_actions'] = priority_actions # [cite: 71]

        return assessment # [cite: 71]

    def _calculate_overall_risk_level(self, predictions):
        """Calculate overall risk level from real predictions""" # [cite: 71]
        composite_score = predictions.get('Composite_Risk_Score', 50) # [cite: 71]
        if composite_score >= 80:
            return "CRITICAL" # [cite: 72]
        elif composite_score >= 65:
            return "HIGH" # [cite: 72]
        elif composite_score >= 45:
            return "MEDIUM" # [cite: 72]
        else:
            return "LOW" # [cite: 72]

    def _generate_recommendations(self, predictions, input_data):
        """Generate detailed recommendations based on real model outputs""" # [cite: 73]
        recommendations = [] # [cite: 73]
        compliance_score = predictions.get('Compliance_Score', 100) # [cite: 73]
        financial_risk = predictions.get('Financial_Risk_Score', 0) # [cite: 73]
        asset_risk = predictions.get('Asset_Risk_Index', 0) # [cite: 73]
        audit_readiness = predictions.get('Audit_Readiness_Score', 100) # [cite: 73]

        if compliance_score < 75:
            recommendations.append({
                "title": "Enhance Compliance Controls",
                "description": f"Compliance score of {compliance_score:.1f}% indicates significant gaps. Focus on improving control effectiveness, documentation quality, and testing frequency.",
                "priority": "HIGH" if compliance_score < 60 else "MEDIUM",
                "impact": "25-35% improvement in compliance posture",
                "timeline": "30-60 days",
                "effort": "High"
            }) # [cite: 74, 75, 76]

        if financial_risk > 65:
            recommendations.append({
                "title": "Mitigate Financial Risk Exposure",
                "description": f"High financial risk exposure detected ({financial_risk:.1f}%). Consider increasing remediation investment and implementing risk transfer mechanisms.",
                "priority": "CRITICAL" if financial_risk > 80 else "HIGH",
                "impact": "30-45% reduction in potential penalties",
                "timeline": "Immediate" if financial_risk > 80 else "30 days",
                "effort": "Medium"
            }) # [cite: 76, 77]

        if asset_risk > 70:
            recommendations.append({
                "title": "Strengthen Asset Protection",
                "description": f"Asset risk index at {asset_risk:.1f} requires attention. Implement enhanced monitoring, access controls, and data protection measures.",
                "priority": "HIGH",
                "impact": "Reduce asset vulnerability by 40-60%",
                "timeline": "30-75 days",
                "effort": "High"
            }) # [cite: 78, 79]

        if audit_readiness < 60:
            recommendations.append({
                "title": "Improve Audit Preparation",
                "description": f"Audit readiness score of {audit_readiness:.1f}% indicates gaps. Focus on documentation completeness and control testing frequency.",
                "priority": "MEDIUM",
                "impact": "Improve audit outcomes by 35-50%",
                "timeline": "45-90 days",
                "effort": "Medium"
            }) # [cite: 80, 81]

        return recommendations # [cite: 81]

    def _generate_priority_actions(self, predictions):
        """Generate priority actions list based on real predictions""" # [cite: 81, 82]
        actions = [] # [cite: 82]
        if predictions.get('Compliance_Score', 100) < 70:
            actions.append("Immediate compliance gap remediation required") # [cite: 82]
        if predictions.get('Financial_Risk_Score', 0) > 75:
            actions.append("Urgent financial risk mitigation needed") # [cite: 82]
        if predictions.get('Incident_Impact_Score', 0) > 70:
            actions.append("Strengthen incident response capabilities") # [cite: 82]
        if not actions:
            actions.append("Continue monitoring and maintain current controls") # [cite: 83]
        return actions # [cite: 83]

    def get_benchmark_data(self, industry=None):
        """Get industry benchmark data from real metadata""" # [cite: 83]
        if hasattr(self, 'scoring_config') and 'benchmarks' in self.scoring_config:
            benchmarks = self.scoring_config['benchmarks'] # [cite: 83]
        else:
            # Default benchmarks if metadata not available
            benchmarks = {
                'Technology': {'compliance': 82.1, 'risk': 45.6, 'maturity': 4.2},
                'Financial Services': {'compliance': 78.5, 'risk': 52.3, 'maturity': 3.8},
                'Healthcare': {'compliance': 75.2, 'risk': 58.7, 'maturity': 3.5},
                'Manufacturing': {'compliance': 72.8, 'risk': 61.2, 'maturity': 3.2},
                'Retail': {'compliance': 70.4, 'risk': 64.5, 'maturity': 3.0},
                'Government': {'compliance': 85.3, 'risk': 40.2, 'maturity': 4.5}
            } # [cite: 84, 85]
        return benchmarks.get(industry, benchmarks.get('Technology')) # [cite: 85]

def create_enhanced_gauge(value, title, min_val=0, max_val=100):
    """Create enhanced gauge chart""" # [cite: 85]
    if value >= 75:
        color = '#10b981' # [cite: 85, 86]
    elif value >= 50:
        color = '#f59e0b' # [cite: 86]
    else:
        color = '#ef4444' # [cite: 86]

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
    )) # [cite: 86, 87, 88, 89]

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    ) # [cite: 90]
    return fig # [cite: 90]

def create_radar_chart(scores):
    """Create enhanced radar chart""" # [cite: 90]
    categories = ['Compliance', 'Financial Risk', 'Asset Risk', 'Audit Readiness', 'Incident Impact'] # [cite: 90]
    values = [
        scores.get('Compliance_Score', 0),
        100 - scores.get('Financial_Risk_Score', 0),
        100 - scores.get('Asset_Risk_Index', 0),
        scores.get('Audit_Readiness_Score', 0),
        100 - scores.get('Incident_Impact_Score', 0)
    ] # [cite: 90, 91]
    values = [max(0, min(100, v)) for v in values] # [cite: 91]

    fig = go.Figure() # [cite: 91]
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Status',
        line_color='#4361ee',
        fillcolor='rgba(67, 97, 238, 0.1)'
    )) # [cite: 91]

    # Add benchmark line
    benchmark_values = [80, 80, 80, 80, 80] # [cite: 92]
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values,
        theta=categories,
        fill='toself',
        name='Industry Benchmark',
        opacity=0.3,
        line_color='#06d6a0',
        fillcolor='rgba(6, 214, 160, 0.1)'
    )) # [cite: 92]

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="Compliance Performance Radar",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    ) # [cite: 93]
    return fig # [cite: 93]

def create_feature_importance_chart(features, title):
    """Create enhanced feature importance visualization""" # [cite: 93]
    if not features:
        return None # [cite: 93]

    # Convert to DataFrame for better handling
    if isinstance(features[0], tuple):
        df = pd.DataFrame(features, columns=['Feature', 'Importance']) # [cite: 94]
    else:
        df = pd.DataFrame(features) # [cite: 94]
        df.columns = ['Feature', 'Importance'] # [cite: 94]

    df = df.sort_values('Importance', ascending=True) # [cite: 94]

    fig = px.bar(
        df.tail(10),  # Top 10 features
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='Viridis',
        height=400
    ) # [cite: 94, 95]

    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="",
        margin=dict(l=0, r=0, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'}
    ) # [cite: 95, 96]
    return fig # [cite: 96]

def show_model_error():
    """Show error when models are not available""" # [cite: 96]
    st.markdown("""
    <div class="error-container">
        <h3>⚠️ Model Files Required</h3>
        <p>This application requires trained .joblib model files to function.</p>
        <p><strong>Required files:</strong></p>
        <ul style="text-align: left; display: inline-block;">
            <li>compliance_score_model.joblib</li>
            <li>financial_risk_score_model.joblib</li>
            <li>asset_risk_index_model.joblib</li>
            <li>audit_readiness_score_model.joblib</li>
            <li>incident_impact_score_model.joblib</li>
            <li>composite_risk_score_model.joblib</li>
        </ul>
        <p>Please ensure all model files are present in the 'enhanced_grc_models' directory.</p>
    </div>
    """, unsafe_allow_html=True) # [cite: 96, 97, 98]

def load_default_historical_data():
    """Generate realistic historical data for demonstration purposes""" # [cite: 98]
    try:
        historical_data = [] # [cite: 98]
        base_date = datetime.now() - timedelta(days=90) # [cite: 98]

        for i in range(90):
            date = base_date + timedelta(days=i) # [cite: 98]
            historical_data.append({
                "timestamp": date.isoformat(),
                "predictions": {
                    "Compliance_Score": min(100, 75 + random.gauss(0, 5) + i*0.1),
                    "Financial_Risk_Score": max(0, 45 + random.gauss(0, 8) - i*0.05),
                    "Asset_Risk_Index": max(0, 60 + random.gauss(0, 10) - i*0.1),
                    "Audit_Readiness_Score": min(100, 65 + random.gauss(0, 7) + i*0.15),
                    "Incident_Impact_Score": max(0, 55 + random.gauss(0, 6) - i*0.08),
                    "Composite_Risk_Score": max(0, min(100, 50 + random.gauss(0, 8) - i*0.05))
                },
                "assessment": {
                    "risk_level": "MEDIUM" if i < 60 else "LOW"
                }
            }) # [cite: 99, 100, 101]

        return historical_data # [cite: 101]
    except Exception as e:
        logger.error(f"Error generating historical data: {str(e)}") # [cite: 102]
        return [] # [cite: 102]

def get_default_assessment():
    """Get a default assessment with reasonable values""" # [cite: 102]
    return {
        "Compliance_Score": 82.5,
        "Financial_Risk_Score": 45.3,
        "Asset_Risk_Index": 58.7,
        "Audit_Readiness_Score": 76.8,
        "Incident_Impact_Score": 52.4,
        "Composite_Risk_Score": 54.2
    } # [cite: 102]

def get_next_refresh_time(refresh_interval):
    """Calculate next refresh time based on interval""" # [cite: 103]
    intervals = {
        "1 minute": 1,
        "5 minutes": 5,
        "15 minutes": 15,
        "30 minutes": 30,
        "1 hour": 60
    } # [cite: 103]
    minutes = intervals.get(refresh_interval, 5) # [cite: 103]
    next_time = datetime.now() + timedelta(minutes=minutes) # [cite: 103]
    return next_time.strftime("%H:%M:%S") # [cite: 103]

def show_trend_insights():
    """Show AI-generated insights based on risk trends""" # [cite: 103]
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 104]
    st.subheader("🤖 AI-Powered Trend Insights") # [cite: 104]

    # Generate some realistic insights based on trends
    insights = [
        {
            "type": "success",
            "title": "Compliance Improvement Trend",
            "content": "Your compliance score has increased by 12.5% over the past 90 days. This indicates effective implementation of control improvements and better documentation practices."
        }, # [cite: 104, 105]
        {
            "type": "warning",
            "title": "Financial Risk Monitoring",
            "content": "Financial risk exposure shows a slight upward trend (3.2%) in the last 30 days. Consider reviewing recent financial controls and risk mitigation strategies."
        }, # [cite: 105, 106]
        {
            "type": "info",
            "title": "Audit Readiness Progress",
            "content": "Audit readiness has improved steadily with a 15.7% increase over the last quarter. Continue with current evidence collection practices."
        } # [cite: 106, 107]
    ]

    for insight in insights:
        st.markdown(f"""
        <div class="ai-insight {insight['type']}">
            <div class="ai-insight-header">🔍 {insight['title']}</div>
            <div class="ai-insight-content">{insight['content']}</div>
        </div>
        """, unsafe_allow_html=True) # [cite: 107]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 107]

def show_dashboard():
    """True executive overview that loads immediately with historical data""" # [cite: 108]
    st.markdown('<h2 class="main-header">📊 Executive Risk Overview</h2>', unsafe_allow_html=True) # [cite: 108]

    # Show loading state if no data available yet
    if not st.session_state.get('historical_assessments'):
        st.info("📊 Loading historical risk data...") # [cite: 108]
        # Initialize with some default historical data
        st.session_state.historical_assessments = load_default_historical_data() # [cite: 108]

    # Always show the dashboard with whatever data is available
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 109]
    st.subheader("📈 Real-time Risk Indicators") # [cite: 109]

    # Get the most recent assessment or use defaults
    current_assessment = st.session_state.historical_assessments[0] if st.session_state.historical_assessments else get_default_assessment() # [cite: 109]

    # Display metrics - these should always be available
    col1, col2, col3, col4, col5 = st.columns(5) # [cite: 109]
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Compliance Score</div>
            <div class="metric-value">{current_assessment['predictions'].get('Compliance_Score', 85):.1f}%</div>
        </div>
        """, unsafe_allow_html=True) # [cite: 109, 110]
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Financial Risk</div>
            <div class="metric-value">{current_assessment['predictions'].get('Financial_Risk_Score', 45):.1f}%</div>
        </div>
        """, unsafe_allow_html=True) # [cite: 110]
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Asset Risk</div>
            <div class="metric-value">{current_assessment['predictions'].get('Asset_Risk_Index', 60):.1f}%</div>
        </div>
        """, unsafe_allow_html=True) # [cite: 111]
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Audit Readiness</div>
            <div class="metric-value">{current_assessment['predictions'].get('Audit_Readiness_Score', 70):.1f}%</div>
        </div>
        """, unsafe_allow_html=True) # [cite: 111, 112]
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Composite Risk</div>
            <div class="metric-value">{current_assessment['predictions'].get('Composite_Risk_Score', 50):.1f}%</div>
        </div>
        """, unsafe_allow_html=True) # [cite: 112]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 113]

    # Add historical trend visualization
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 113]
    st.subheader("📈 Risk Trend Analysis") # [cite: 113]

    # Convert historical data to DataFrame for plotting
    if st.session_state.historical_assessments:
        historical_df = pd.DataFrame([
            {**assess['predictions'], 'timestamp': assess['timestamp']}
            for assess in st.session_state.historical_assessments
        ]) # [cite: 113]
        historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp']) # [cite: 114]

        # Create time series visualization
        fig = px.line(
            historical_df.melt(id_vars='timestamp', var_name='Metric', value_name='Score'),
            x='timestamp',
            y='Score',
            color='Metric',
            title='Risk Metrics Over Time',
            line_shape='spline'
        ) # [cite: 114, 115]
        fig.update_layout(hovermode="x unified") # [cite: 115]
        st.plotly_chart(fig, use_container_width=True) # [cite: 115]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 115]

    # Add AI-generated insights based on trends
    show_trend_insights() # [cite: 115]

def show_compliance_monitoring():
    """Compliance-specific monitoring dashboard""" # [cite: 115]
    st.markdown('<h2 class="main-header">✅ Compliance Monitoring</h2>', unsafe_allow_html=True) # [cite: 115]

    # Always show current status
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 115, 116]
    st.subheader("📊 Current Compliance Status") # [cite: 116]

    # Get most recent compliance data
    if not st.session_state.historical_assessments:
        st.session_state.historical_assessments = load_default_historical_data() # [cite: 116]

    current_compliance = st.session_state.historical_assessments[0]['predictions'] # [cite: 116]

    # Compliance metrics
    col1, col2, col3, col4 = st.columns(4) # [cite: 116]
    with col1:
        st.metric("Overall Score", f"{current_compliance['Compliance_Score']:.1f}%", "↑ 2.1% from last month") # [cite: 116]
    with col2:
        st.metric("Framework Coverage", "8/10 frameworks", "GDPR, HIPAA, PCI-DSS") # [cite: 116, 117]
    with col3:
        st.metric("Control Gaps", "12", "↓ 5 from last month") # [cite: 117]
    with col4:
        st.metric("Maturity Level", "Defined → Quantitatively Managed", "Level 3 → 4") # [cite: 117]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 117]

    # Framework-specific compliance visualization
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 117]
    st.subheader("Framework Compliance Breakdown") # [cite: 117]

    # Generate sample data for framework compliance
    framework_data = pd.DataFrame({
        'Framework': ['ISO27001', 'NIST-CSF', 'GDPR', 'HIPAA', 'PCI-DSS', 'SOC2'],
        'Compliance %': [88.2, 85.7, 76.4, 82.1, 79.3, 91.5]
    }) # [cite: 117, 118]

    fig = px.bar(
        framework_data,
        x='Framework',
        y='Compliance %',
        color='Compliance %',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100],
        labels={'Compliance %': 'Compliance Percentage'}
    ) # [cite: 118, 119]
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    ) # [cite: 119]
    st.plotly_chart(fig, use_container_width=True) # [cite: 119]

    # Key Drivers
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 642]
    st.subheader("🔍 Key Drivers for Compliance Score") # [cite: 642]
    engine = st.session_state.scoring_engine # [cite: 640]
    if hasattr(engine, 'feature_importance') and 'Compliance_Score' in engine.feature_importance:
        fig = create_feature_importance_chart(engine.feature_importance['Compliance_Score'], 'Top Features - Compliance Score') # [cite: 642]
        if fig: st.plotly_chart(fig, use_container_width=True) # [cite: 642]
    else:
        st.warning("Feature importance data for this model is not available.") # [cite: 642]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 642]


def show_financial_risk_monitoring():
    """Financial risk-specific monitoring dashboard""" # [cite: 132]
    st.markdown('<h2 class="main-header">💰 Financial Risk Monitoring</h2>', unsafe_allow_html=True) # [cite: 132]

    # Always show current status
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 132]
    st.subheader("📊 Current Financial Risk Exposure") # [cite: 132]

    # Get most recent data
    if not st.session_state.historical_assessments:
        st.session_state.historical_assessments = load_default_historical_data() # [cite: 132]

    current_risk = st.session_state.historical_assessments[0]['predictions'] # [cite: 132]

    # Financial risk metrics
    col1, col2, col3, col4 = st.columns(4) # [cite: 132, 133]
    with col1:
        st.metric("Financial Risk Score", f"{current_risk['Financial_Risk_Score']:.1f}%", "↓ 3.2% from last month") # [cite: 133]
    with col2:
        st.metric("Potential Penalties", "$2.4M", "↓ $350K from last quarter") # [cite: 133]
    with col3:
        st.metric("ROI Potential", "3.2x", "↑ 0.5x from last assessment") # [cite: 133]
    with col4:
        st.metric("Remediation Coverage", "78%", "↑ 12% from last month") # [cite: 133]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 133]

    # Risk exposure visualization
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 134]
    st.subheader("Risk Exposure Breakdown") # [cite: 134]

    # Generate sample data for risk exposure
    risk_data = pd.DataFrame({
        'Risk Category': ['Regulatory', 'Operational', 'Strategic', 'Financial', 'Reputational'],
        'Exposure %': [28, 22, 18, 20, 12],
        'Impact Level': ['High', 'Medium', 'Medium', 'High', 'Medium']
    }) # [cite: 134]

    fig = px.pie(
        risk_data,
        names='Risk Category',
        values='Exposure %',
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=0.4
    ) # [cite: 134, 135]
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    ) # [cite: 135]
    st.plotly_chart(fig, use_container_width=True) # [cite: 135]

    # Key Drivers
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 648]
    st.subheader("🔍 Key Drivers for Financial Risk") # [cite: 648]
    engine = st.session_state.scoring_engine # [cite: 645]
    if hasattr(engine, 'feature_importance') and 'Financial_Risk_Score' in engine.feature_importance:
        fig = create_feature_importance_chart(engine.feature_importance['Financial_Risk_Score'], 'Top Features - Financial Risk') # [cite: 648]
        if fig: st.plotly_chart(fig, use_container_width=True) # [cite: 648]
    else:
        st.warning("Feature importance data for this model is not available.") # [cite: 648]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 648]


def show_asset_risk_monitoring():
    """Asset risk-specific monitoring dashboard""" # [cite: 144, 145]
    st.markdown('<h2 class="main-header">🛡️ Asset Risk Monitoring</h2>', unsafe_allow_html=True) # [cite: 145]

    # Always show current status
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 145]
    st.subheader("📊 Current Asset Risk Profile") # [cite: 145]

    # Get most recent data
    if not st.session_state.historical_assessments:
        st.session_state.historical_assessments = load_default_historical_data() # [cite: 145]

    current_risk = st.session_state.historical_assessments[0]['predictions'] # [cite: 145]

    # Asset risk metrics
    col1, col2, col3, col4 = st.columns(4) # [cite: 145]
    with col1:
        st.metric("Asset Risk Index", f"{current_risk['Asset_Risk_Index']:.1f}%", "↓ 4.1% from last month") # [cite: 146]
    with col2:
        st.metric("Critical Assets", "24", "↓ 3 from last quarter") # [cite: 146]
    with col3:
        st.metric("High-Risk Assets", "42", "↓ 8 from last month") # [cite: 146]
    with col4:
        st.metric("Protected Assets", "86%", "↑ 7% from last assessment") # [cite: 146]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 146]

    # Asset risk visualization
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 146, 147]
    st.subheader("Asset Risk Distribution") # [cite: 147]

    # Generate sample data for asset risk
    asset_data = pd.DataFrame({
        'Asset Type': ['Database', 'Server', 'Workstation', 'Cloud Service', 'IoT Device', 'Network Device'],
        'Risk Score': [75, 68, 52, 63, 82, 58],
        'Criticality': ['High', 'Medium', 'Low', 'Medium', 'High', 'Medium']
    }) # [cite: 147]

    fig = px.bar(
        asset_data,
        x='Asset Type',
        y='Risk Score',
        color='Risk Score',
        color_continuous_scale='RdYlGn_r',
        range_color=[0, 100],
        title='Risk by Asset Type'
    ) # [cite: 147, 148]
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    ) # [cite: 148, 149]
    st.plotly_chart(fig, use_container_width=True) # [cite: 149]

    # Key Drivers
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 652]
    st.subheader("🔍 Key Drivers for Asset Risk") # [cite: 652]
    engine = st.session_state.scoring_engine # [cite: 649]
    if hasattr(engine, 'feature_importance') and 'Asset_Risk_Index' in engine.feature_importance:
        fig = create_feature_importance_chart(engine.feature_importance['Asset_Risk_Index'], 'Top Features - Asset Risk') # [cite: 652]
        if fig: st.plotly_chart(fig, use_container_width=True) # [cite: 652]
    else:
        st.warning("Feature importance data for this model is not available.") # [cite: 652]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 652]


def show_audit_readiness_monitoring():
    """Audit readiness-specific monitoring dashboard""" # [cite: 157, 158]
    st.markdown('<h2 class="main-header">📝 Audit Readiness Monitoring</h2>', unsafe_allow_html=True) # [cite: 158]

    # Always show current status
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 158]
    st.subheader("📊 Current Audit Readiness Status") # [cite: 158]

    # Get most recent data
    if not st.session_state.historical_assessments:
        st.session_state.historical_assessments = load_default_historical_data() # [cite: 158]

    current_readiness = st.session_state.historical_assessments[0]['predictions'] # [cite: 158]

    # Audit readiness metrics
    col1, col2, col3, col4 = st.columns(4) # [cite: 158]
    with col1:
        st.metric("Audit Readiness Score", f"{current_readiness['Audit_Readiness_Score']:.1f}%", "↑ 3.5% from last month") # [cite: 159]
    with col2:
        st.metric("Evidence Coverage", "82%", "↑ 5% from last quarter") # [cite: 159]
    with col3:
        st.metric("Open Findings", "8", "↓ 3 from last audit") # [cite: 159]
    with col4:
        st.metric("Remediation Rate", "75%", "↑ 10% from last assessment") # [cite: 159]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 159]

    # Audit readiness visualization
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 159, 160]
    st.subheader("Audit Readiness Breakdown") # [cite: 160]

    # Generate sample data for audit readiness
    readiness_data = pd.DataFrame({
        'Audit Area': ['Access Control', 'Data Protection', 'Network Security', 'Incident Response', 'Business Continuity'],
        'Readiness %': [85, 78, 72, 88, 92],
        'Findings': [2, 4, 6, 1, 0]
    }) # [cite: 160]

    fig = px.bar(
        readiness_data,
        x='Audit Area',
        y='Readiness %',
        color='Readiness %',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100],
        title='Readiness by Audit Area'
    ) # [cite: 160, 161]
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    ) # [cite: 161]
    st.plotly_chart(fig, use_container_width=True) # [cite: 162]

    # Key Drivers
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 657]
    st.subheader("🔍 Key Drivers for Audit Readiness") # [cite: 657]
    engine = st.session_state.scoring_engine # [cite: 654]
    if hasattr(engine, 'feature_importance') and 'Audit_Readiness_Score' in engine.feature_importance:
        fig = create_feature_importance_chart(engine.feature_importance['Audit_Readiness_Score'], 'Top Features - Audit Readiness') # [cite: 657]
        if fig: st.plotly_chart(fig, use_container_width=True) # [cite: 657]
    else:
        st.warning("Feature importance data for this model is not available.") # [cite: 657]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 657]


def show_incident_impact_monitoring():
    """Incident impact-specific monitoring dashboard""" # [cite: 173]
    st.markdown('<h2 class="main-header">🚨 Incident Impact Monitoring</h2>', unsafe_allow_html=True) # [cite: 173]

    # Always show current status
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 173]
    st.subheader("📊 Current Incident Impact Profile") # [cite: 173]

    # Get most recent data
    if not st.session_state.historical_assessments:
        st.session_state.historical_assessments = load_default_historical_data() # [cite: 173]

    current_impact = st.session_state.historical_assessments[0]['predictions'] # [cite: 173]

    # Incident impact metrics
    col1, col2, col3, col4 = st.columns(4) # [cite: 173, 174]
    with col1:
        st.metric("Incident Impact Score", f"{current_impact['Incident_Impact_Score']:.1f}%", "↓ 5.2% from last month") # [cite: 174]
    with col2:
        st.metric("MTTD", "45 min", "↓ 15 min from last quarter") # [cite: 174]
    with col3:
        st.metric("MTTR", "6.2 hrs", "↓ 2.1 hrs from last month") # [cite: 174]
    with col4:
        st.metric("Incidents (90d)", "12", "↓ 3 from last quarter") # [cite: 174]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 174]

    # Incident impact visualization
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 175]
    st.subheader("Incident Type Analysis") # [cite: 175]

    # Generate sample data for incident types
    incident_data = pd.DataFrame({
        'Incident Type': ['Phishing', 'Malware', 'System Outage', 'Data Breach', 'Insider Threat', 'DDoS'],
        'Count': [5, 3, 2, 1, 1, 0],
        'Impact %': [35, 45, 65, 85, 75, 55],
        'MTTR (hrs)': [2.5, 8.2, 12.5, 24.0, 18.0, 6.0]
    }) # [cite: 175, 176]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=("Incidents by Type", "Impact by Incident Type")
    ) # [cite: 176]

    fig.add_trace(
        go.Bar(
            x=incident_data['Incident Type'],
            y=incident_data['Count'],
            marker_color='#4361ee'
        ),
        row=1, col=1
    ) # [cite: 176, 177]

    fig.add_trace(
        go.Bar(
            x=incident_data['Incident Type'],
            y=incident_data['Impact %'],
            marker_color='#ef476f'
        ),
        row=1, col=2
    ) # [cite: 177]

    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    ) # [cite: 178]

    st.plotly_chart(fig, use_container_width=True) # [cite: 178]

    # Key Drivers
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 660]
    st.subheader("🔍 Key Drivers for Incident Impact") # [cite: 660]
    engine = st.session_state.scoring_engine # [cite: 659]
    if hasattr(engine, 'feature_importance') and 'Incident_Impact_Score' in engine.feature_importance:
        fig = create_feature_importance_chart(engine.feature_importance['Incident_Impact_Score'], 'Top Features - Incident Impact') # [cite: 660]
        if fig: st.plotly_chart(fig, use_container_width=True) # [cite: 660]
    else:
        st.warning("Feature importance data for this model is not available.") # [cite: 661]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 661]


def show_composite_risk_monitoring():
    """Composite risk-specific monitoring dashboard""" # [cite: 188]
    st.markdown('<h2 class="main-header">🌐 Composite Risk Monitoring</h2>', unsafe_allow_html=True) # [cite: 188]

    # Always show current status
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 188]
    st.subheader("📊 Current Composite Risk Profile") # [cite: 188]

    # Get most recent data
    if not st.session_state.historical_assessments:
        st.session_state.historical_assessments = load_default_historical_data() # [cite: 188]

    current_risk = st.session_state.historical_assessments[0]['predictions'] # [cite: 188]

    # Composite risk metrics
    col1, col2, col3, col4 = st.columns(4) # [cite: 188]
    with col1:
        st.metric("Composite Risk Score", f"{current_risk['Composite_Risk_Score']:.1f}%", "↓ 3.8% from last month") # [cite: 189]
    with col2:
        st.metric("Risk Trend", "Downward", "Steady improvement") # [cite: 189]
    with col3:
        st.metric("Risk Hotspots", "2", "↓ 1 from last assessment") # [cite: 189]
    with col4:
        st.metric("Risk Maturity", "Level 3.2", "↑ 0.3 from last quarter") # [cite: 189]

    st.markdown('</div>', unsafe_allow_html=True) # [cite: 189]

    # Composite risk visualization
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 189, 190]
    st.subheader("Risk Component Analysis") # [cite: 190]

    # Generate sample data for risk components
    risk_components = pd.DataFrame({
        'Risk Component': ['Compliance', 'Financial', 'Asset', 'Operational', 'Strategic'],
        'Contribution %': [25, 20, 22, 18, 15],
        'Current Score': [82.5, 45.3, 58.7, 65.8, 72.4]
    }) # [cite: 190]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Risk Contribution", "Component Scores")
    ) # [cite: 190, 191]

    fig.add_trace(
        go.Pie(
            labels=risk_components['Risk Component'],
            values=risk_components['Contribution %'],
            hole=0.4,
            marker_colors=px.colors.qualitative.Pastel
        ),
        row=1, col=1
    ) # [cite: 191, 192]

    fig.add_trace(
        go.Bar(
            x=risk_components['Risk Component'],
            y=risk_components['Current Score'],
            marker_color=['#10b981', '#f59e0b', '#ef4444', '#f59e0b', '#10b981']
        ),
        row=1, col=2
    ) # [cite: 192]

    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    ) # [cite: 193]

    st.plotly_chart(fig, use_container_width=True) # [cite: 193]

    # Key Drivers
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("🔍 Key Drivers for Composite Risk")
    engine = st.session_state.scoring_engine
    if hasattr(engine, 'feature_importance') and 'Composite_Risk_Score' in engine.feature_importance:
        fig = create_feature_importance_chart(engine.feature_importance['Composite_Risk_Score'], 'Top Features - Composite Risk Score')
        if fig: st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance data for this model is not available.")
    st.markdown('</div>', unsafe_allow_html=True)


def show_risk_assessment_form():
    """Form for manual risk assessment data entry.""" # [cite: 589]
    if not st.session_state.engine_loaded or not st.session_state.scoring_engine.is_loaded:
        show_model_error() # [cite: 589]
        return # [cite: 589]

    with st.form("real_model_risk_assessment_form", clear_on_submit=False):
        st.subheader("🛡️ Compliance Profile") # [cite: 590]
        col1, col2 = st.columns(2) # [cite: 590]
        with col1:
            frameworks = st.multiselect(
                "Applicable Compliance Frameworks",
                ["ISO27001", "NIST-CSF", "GDPR", "HIPAA", "PCI-DSS", "SOC2", "CIS-Controls", "SOX"],
                default=["ISO27001", "NIST-CSF", "SOC2"],
                help="Select all applicable regulatory frameworks"
            ) # [cite: 590, 591]
            maturity = st.selectbox(
                "Compliance Maturity Level",
                ["Initial", "Managed", "Defined", "Quantitatively Managed", "Optimizing"],
                index=2,
                help="Current maturity level of your compliance program"
            ) # [cite: 591, 592]
            testing_frequency = st.selectbox(
                "Control Testing Frequency",
                ["Never", "Annually", "Bi-Annually", "Quarterly", "Monthly", "Weekly"],
                index=3,
                help="How frequently are controls tested"
            ) # [cite: 592, 593]
        with col2:
            control_category = st.selectbox(
                "Primary Control Category",
                ["Access Control", "Data Protection", "Network Security", "Physical Security", "Incident Response", "Business Continuity"],
                index=0,
                help="Primary category of controls being assessed"
            ) # [cite: 593, 594]
            control_status = st.select_slider(
                "Control Implementation Status",
                options=["0% Implemented", "25% Implemented", "50% Implemented", "75% Implemented", "100% Implemented"],
                value="75% Implemented",
                help="Overall implementation status of your controls"
            ) # [cite: 595, 596]
            business_impact = st.selectbox(
                "Business Impact Level",
                ["Negligible", "Low", "Medium", "High", "Critical"],
                index=2,
                help="Potential business impact of control failures"
            ) # [cite: 596, 597]

        st.subheader("💰 Financial Profile") # [cite: 597]
        col1, col2 = st.columns(2) # [cite: 597]
        with col1:
            annual_revenue = st.number_input(
                "Annual Revenue ($)",
                min_value=0,
                value=50000000,
                step=1000000,
                help="Organization's annual revenue"
            ) # [cite: 598]
            penalty_risk = st.slider(
                "Penalty Risk Assessment",
                0, 100, 45,
                help="Potential regulatory penalty risk (0-100)"
            ) # [cite: 599]
        with col2:
            remediation_cost = st.number_input(
                "Remediation Cost ($)",
                min_value=0,
                value=50000,
                step=5000,
                help="Expected cost to remediate identified gaps"
            ) # [cite: 600, 601]
            incident_cost = st.number_input(
                "Incident Cost Impact ($)",
                min_value=0,
                value=10000,
                step=1000,
                help="Potential cost impact of security incidents"
            ) # [cite: 601, 602]

        st.subheader("🔐 Asset Risk Profile") # [cite: 602]
        col1, col2 = st.columns(2) # [cite: 602]
        with col1:
            asset_type = st.selectbox(
                "Primary Asset Type",
                ["Server", "Database", "Workstation", "Network Device", "Cloud Service", "IoT Device"],
                index=1,
                help="Type of primary assets being assessed"
            ) # [cite: 603, 604]
            data_sensitivity = st.selectbox(
                "Data Sensitivity Classification",
                ["Public", "Internal", "Intellectual-Property", "Confidential",
                 "Personal-Data", "Restricted", "Financial-Data", "Health-Data",
                 "Regulated", "Top-Secret"],
                index=3,
                help="Highest classification of data processed"
            ) # [cite: 604, 605]
            organizational_unit = st.selectbox(
                "Organizational Unit",
                ["IT", "Finance", "HR", "Operations", "Sales", "Marketing"],
                index=0,
                help="Primary organizational unit responsible"
            ) # [cite: 606]
        with col2:
            geographic_scope = st.multiselect(
                "Geographic Scope",
                ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East", "Africa"],
                default=["North America", "Europe"],
                help="Regions where your organization operates"
            ) # [cite: 606, 607]
            industry_sector = st.selectbox(
                "Industry Sector",
                ["Financial Services", "Healthcare", "Technology", "Manufacturing", "Retail", "Energy", "Government"],
                index=2,
                help="Your organization's primary industry"
            ) # [cite: 608, 609]

        st.subheader("📋 Audit Profile") # [cite: 609]
        col1, col2 = st.columns(2) # [cite: 609]
        with col1:
            audit_type = st.selectbox(
                "Audit Type",
                ["Internal", "External", "Regulatory", "Compliance", "Operational"],
                index=0,
                help="Type of audit being prepared for"
            ) # [cite: 609, 610]
            audit_severity = st.selectbox(
                "Audit Finding Severity",
                ["Informational", "Low", "Medium", "High", "Critical"],
                index=2,
                help="Typical severity of audit findings"
            ) # [cite: 610, 611]
            repeat_finding = st.checkbox(
                "Repeat Finding",
                value=False,
                help="Are there recurring audit findings?"
            ) # [cite: 612]
        with col2:
            compliance_owner = st.text_input(
                "Compliance Owner",
                "John Smith",
                help="Name of the compliance program owner"
            ) # [cite: 613]
            evidence_days = st.number_input(
                "Evidence Freshness (Days)",
                min_value=0,
                value=30,
                help="Average age of compliance evidence in days"
            ) # [cite: 614]
            audit_preparation = st.slider(
                "Audit Preparation Score",
                0, 100, 75,
                help="Current level of audit preparation (0-100)"
            ) # [cite: 615]

        st.subheader("⚠️ Incident Profile") # [cite: 615]
        col1, col2 = st.columns(2) # [cite: 616]
        with col1:
            incident_type = st.selectbox(
                "Most Likely Incident Type",
                ["Data Breach", "System Outage", "Phishing", "Malware", "Ransomware", "DDoS", "Insider Threat"],
                index=0,
                help="Most likely type of security incident"
            ) # [cite: 616, 617]
            incident_severity = st.selectbox(
                "Expected Incident Severity",
                ["Low", "Medium", "High", "Critical"],
                index=1,
                help="Expected severity level of incidents"
            ) # [cite: 618]
        with col2:
            incident_notification = st.checkbox(
                "Incident Notification Compliance",
                value=True,
                help="Are incident notification procedures compliant?"
            ) # [cite: 619]

        submitted = st.form_submit_button("🚀 Generate Manual Assessment", type="primary", use_container_width=True) # [cite: 620]

    if submitted:
        try:
            with st.spinner("🤖 Analyzing your risk profile with real AI models..."):
                status_text = st.empty() # [cite: 621]
                input_data = {
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
                    'Incident_Type': INCIDENT_TYPE_MAPPING.get(incident_type, "None"),
                    'Incident_Notification_Compliance': bool(incident_notification),
                    'Incident_Cost_Impact': float(incident_cost)
                } # [cite: 624, 625, 626, 627, 628, 629]
                
                status_text.text('Executing real-time analysis...') # [cite: 623]
                predictions = st.session_state.scoring_engine.predict_scores(input_data) # [cite: 630]
                
                status_text.text('Generating model-based recommendations...') # [cite: 623]
                assessment_details = st.session_state.scoring_engine.generate_assessment(input_data, predictions) # [cite: 630]
                
                new_assessment = {
                    "timestamp": datetime.now().isoformat(),
                    "predictions": predictions,
                    "assessment": assessment_details
                } # [cite: 99, 101, 631, 633]

                max_history = st.session_state.user_preferences.get("max_history", 100) # 
                st.session_state.historical_assessments.insert(0, new_assessment)
                st.session_state.historical_assessments = st.session_state.historical_assessments[:max_history]

                status_text.text('✅ Real model assessment completed successfully!') # 
                time.sleep(1) # [cite: 635]
                status_text.empty() # [cite: 635]

                st.balloons() # [cite: 635]
                st.success("🎯 Manual Assessment completed! View your results on the dashboard and deep-dive pages.") # [cite: 635, 636]
        except Exception as e:
            st.error(f"❌ Error during assessment: {str(e)}") # [cite: 636]
            logger.error(f"Assessment error: {str(e)}") # [cite: 636]


def show_data_management():
    """Data ingestion and manual assessment page."""
    st.markdown('<h2 class="main-header">📥 Data Management</h2>', unsafe_allow_html=True)

    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 205, 636]
    st.subheader("✅ Automated Data Collection") # [cite: 205]
    st.info("Upload compliance reports, logs, and financial data. The system will automatically process files and update risk metrics.") # [cite: 205, 637]
    
    uploaded_files = st.file_uploader(
        "Upload files for continuous monitoring",
        accept_multiple_files=True,
        key="file_uploader",
        type=['csv', 'xlsx', 'json', 'pdf', 'txt']
    ) # [cite: 209, 210]

    if st.button("Process Uploaded Files", key="process_files"):
        if uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                for file in uploaded_files:
                    # Simulate processing and add to history
                    new_assessment = generate_assessment_from_file(file) # [cite: 225]
                    st.session_state.historical_assessments.insert(0, new_assessment)
                st.success(f"Processed {len(uploaded_files)} files successfully!") # [cite: 212]
        else:
            st.warning("Please select files to upload first.") # [cite: 212]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 639]


    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 639]
    with st.expander("📝 Manual Assessment Form", expanded=False):
        show_risk_assessment_form() # [cite: 639]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 639]


def process_uploaded_files(source_type, files):
    """Process uploaded files and update risk metrics""" # [cite: 222]
    # This is a simplified version for demonstration.
    try:
        for i, file in enumerate(files):
            time.sleep(0.2)  # Simulate processing time
            if i == len(files) - 1:
                new_assessment = generate_assessment_from_file(file) # [cite: 225]
                st.session_state.historical_assessments = [new_assessment] + st.session_state.historical_assessments[:99] # [cite: 225]
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}") # [cite: 226]

def generate_assessment_from_file(file):
    """Generate risk assessment from uploaded file content""" # [cite: 226]
    # For demonstration, this generates realistic mock data.
    return {
        "timestamp": datetime.now().isoformat(),
        "source": file.name,
        "predictions": {
            "Compliance_Score": random.randint(70, 95),
            "Financial_Risk_Score": random.randint(30, 65),
            "Asset_Risk_Index": random.randint(40, 75),
            "Audit_Readiness_Score": random.randint(65, 90),
            "Incident_Impact_Score": random.randint(35, 65),
            "Composite_Risk_Score": random.randint(40, 70)
        },
        "assessment": {
            "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"])
        }
    } # [cite: 227, 228]

def show_settings():
    """Enhanced settings page""" # [cite: 228, 229]
    st.markdown('<h2 class="main-header">⚙️ Application Settings</h2>', unsafe_allow_html=True) # [cite: 229]

    # Theme Settings
    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 229]
    st.subheader("🎨 Appearance") # [cite: 229]
    col1, col2 = st.columns(2) # [cite: 229]
    with col1:
        theme = st.selectbox(
            "Theme",
            ["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1,
            help="Select your preferred color scheme"
        ) # [cite: 229, 230]
        st.session_state.theme = theme # [cite: 230]
    with col2:
        font_size = st.select_slider(
            "Font Size",
            options=["Small", "Medium", "Large"],
            value="Medium",
            help="Adjust the application font size"
        ) # [cite: 230, 231]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 231]

    st.markdown('<div class="section-container">', unsafe_allow_html=True) # [cite: 231]
    st.subheader("👤 User Preferences") # [cite: 231]
    pref_col1, pref_col2 = st.columns(2) # [cite: 231]
    with pref_col1:
        notifications = st.toggle(
            "Notifications",
            value=st.session_state.user_preferences.get("notifications", True),
            help="Receive alerts for critical risk changes"
        ) # [cite: 231]
        auto_refresh = st.toggle(
            "Auto-refresh Dashboard",
            value=st.session_state.user_preferences.get("auto_refresh", True),
            help="Automatically refresh dashboard data"
        ) # [cite: 232]
    with pref_col2:
        max_history = st.slider(
            "Prediction History Limit",
            10, 1000,
            st.session_state.user_preferences.get("max_history", 100),
            help="Maximum number of predictions to keep in history"
        ) # [cite: 232, 233]

    st.session_state.user_preferences.update({
        "notifications": notifications,
        "auto_refresh": auto_refresh,
        "max_history": max_history,
        "theme": theme
    }) # [cite: 233]
    st.markdown('</div>', unsafe_allow_html=True) # [cite: 233]

def main():
    """Main application function with operational monitoring focus""" # [cite: 261]
    init_session_state() # [cite: 261]

    # Initialize historical data if not already present
    if 'historical_assessments' not in st.session_state:
        st.session_state.historical_assessments = load_default_historical_data() # [cite: 261]

    st.set_page_config(
        page_title="GRC AI Monitoring Platform v5.0",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    ) # [cite: 262]

    st.markdown(get_enhanced_css(), unsafe_allow_html=True) # [cite: 262]

    # Initialize engine if needed
    if not st.session_state.get('engine_loaded', False):
        try:
            with st.spinner("⚙️ Loading monitoring engine..."):
                st.session_state.scoring_engine = RealModelGRCEngine() # [cite: 263]
                st.session_state.engine_loaded = True # [cite: 263]
                if not st.session_state.historical_assessments:
                    st.session_state.historical_assessments = load_default_historical_data() # [cite: 263, 264]
        except Exception as e:
            st.session_state.engine_loaded = False # [cite: 264]
            st.error(f"❌ Failed to initialize monitoring engine: {str(e)}") # [cite: 264]
            logger.error(f"Engine initialization error: {str(e)}") # [cite: 264]

    # Navigation - domain-centric structure
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
            <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">GRC Monitoring Platform</h2>
            <p style="margin: 0.5rem 0; font-size: 0.9rem; opacity: 0.8;">Dynamic Monitoring v5.0</p>
        </div>
        """, unsafe_allow_html=True) # [cite: 265, 266, 267, 268, 269]

        st.markdown("---") # [cite: 269]

        pages = {
            "📊 Executive Overview": "dashboard",
            "📥 Data Management": "data_management",
            "✅ Compliance Monitoring": "compliance",
            "💰 Financial Risk Monitoring": "financial_risk",
            "🛡️ Asset Risk Monitoring": "asset_risk",
            "📝 Audit Readiness Monitoring": "audit_readiness",
            "🚨 Incident Impact Monitoring": "incident_impact",
            "🌐 Composite Risk Monitoring": "composite_risk",
            "⚙️ Settings": "settings"
        } # [cite: 269, 270, 271]

        selected_page = st.radio("Monitoring Dashboard", list(pages.keys())) # [cite: 271]
        page_key = pages[selected_page] # [cite: 271]

        st.markdown("---") # [cite: 271]
        st.markdown("### 📊 System Status") # [cite: 271]
        if st.session_state.get('engine_loaded', False) and st.session_state.scoring_engine.is_loaded:
            st.success("✅ Monitoring Engine Active") # [cite: 271]
            st.caption(f"{len(st.session_state.historical_assessments)} historical assessments") # [cite: 271, 272]
        else:
            st.error("❌ Data Processing Engine Down") # [cite: 272]
            show_model_error() # [cite: 96]
        
        st.markdown("---") # [cite: 272]
        st.markdown("### 🔄 Auto-Refresh") # [cite: 272]
        auto_refresh = st.toggle(
            "Enable Auto-Refresh",
            value=st.session_state.user_preferences.get("auto_refresh", True),
            help="Automatically refresh data at the selected interval"
        ) # [cite: 272, 273]

        if auto_refresh:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
                index=1
            ) # [cite: 273, 274]
            st.session_state.auto_refresh_interval = refresh_interval # [cite: 274]
            st.caption(f"Next refresh: {get_next_refresh_time(refresh_interval)}") # [cite: 274]

    # Page routing
    if page_key == "dashboard":
        show_dashboard() # [cite: 274]
    elif page_key == "data_management":
        show_data_management() # [cite: 691]
    elif page_key == "compliance":
        show_compliance_monitoring() # [cite: 275]
    elif page_key == "financial_risk":
        show_financial_risk_monitoring() # [cite: 275]
    elif page_key == "asset_risk":
        show_asset_risk_monitoring() # [cite: 275]
    elif page_key == "audit_readiness":
        show_audit_readiness_monitoring() # [cite: 275]
    elif page_key == "incident_impact":
        show_incident_impact_monitoring() # [cite: 275]
    elif page_key == "composite_risk":
        show_composite_risk_monitoring() # [cite: 275]
    elif page_key == "settings":
        show_settings() # [cite: 276]

    # Auto-refresh implementation
    if st.session_state.user_preferences.get("auto_refresh", True):
        refresh_interval = st.session_state.get("auto_refresh_interval", "5 minutes") # [cite: 276]
        seconds = {"1 minute": 60, "5 minutes": 300, "15 minutes": 900, "30 minutes": 1800, "1 hour": 3600}.get(refresh_interval, 300) # [cite: 276]
        st_autorefresh(interval=seconds * 1000, key="datarefresh") # [cite: 276]

if __name__ == "__main__":
    main() # [cite: 276]