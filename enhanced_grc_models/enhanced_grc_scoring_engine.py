#!/usr/bin/env python3
"""
Enhanced Production GRC Scoring Engine
Enterprise-grade scoring with comprehensive error handling and optimization
Generated: 2025-09-08 12:27:01
Compatible with scikit-learn 1.7.1
"""
import numpy as np
import pandas as pd
import joblib
import json
import os
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GRCScoreEngine")

class EnhancedGRCScoreEngine:
    """Production-ready GRC scoring engine with enterprise features"""
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or os.path.dirname(os.path.abspath(__file__))
        self.models = {}
        self.feature_info = {}
        self.scoring_config = {}
        self.model_metadata = {}
        self._load_models_and_metadata()
    
    def _load_models_and_metadata(self):
        """Load all models and metadata"""
        try:
            # Load models
            model_files = {
                'Compliance_Score': 'compliance_score_model.joblib',
                'Financial_Risk_Score': 'financial_risk_score_model.joblib',
                'Asset_Risk_Index': 'asset_risk_index_model.joblib',
                'Audit_Readiness_Score': 'audit_readiness_score_model.joblib',
                'Incident_Impact_Score': 'incident_impact_score_model.joblib',
                'Composite_Risk_Score': 'composite_risk_score_model.joblib'
            }
            for score_name, filename in model_files.items():
                model_path = os.path.join(self.model_dir, filename)
                if os.path.exists(model_path):
                    self.models[score_name] = joblib.load(model_path)
                    logger.info(f"Loaded {score_name} model")
                else:
                    logger.error(f"Model file not found: {model_path}")
            
            # Load metadata files
            metadata_files = {
                'feature_info.json': 'feature_info',
                'scoring_config.json': 'scoring_config',
                'model_metrics.json': 'model_metadata'
            }
            for filename, attr_name in metadata_files.items():
                file_path = os.path.join(self.model_dir, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        setattr(self, attr_name, json.load(f))
                else:
                    logger.error(f"Metadata file not found: {file_path}")
            
            logger.info("GRC Scoring Engine initialized successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def preprocess_input(self, data):
        """Enhanced input preprocessing with validation"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Validate required features
        required_features = self.feature_info.get('feature_columns', [])
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            # Fill with defaults for missing features
            for feature in missing_features:
                if feature in self.feature_info.get('numerical_columns', []):
                    df[feature] = 0.0
                elif feature in self.feature_info.get('boolean_columns', []):
                    df[feature] = False
                elif feature in self.feature_info.get('ordinal_features', []):
                    # Use the first category as a safe default for ordinal features
                    try:
                        ordinal_idx = self.feature_info.get('ordinal_features', []).index(feature)
                        default_value = self.feature_info['ordinal_categories'][ordinal_idx][0]
                        df[feature] = default_value
                    except (ValueError, IndexError, KeyError):
                        df[feature] = 'Unknown'  # Fallback
                else:
                    df[feature] = 'Unknown'  # For one-hot encoded features
            logger.warning(f"Missing features filled with defaults: {missing_features}")
        
        # Create derived features
        if 'Annual_Revenue' in df.columns:
            # Handle cases where Annual_Revenue might be zero
            safe_revenue = df['Annual_Revenue'].replace(0, 1).fillna(1)
            if 'Penalty_Risk_Assessment' in df.columns:
                df['Risk_Exposure_Ratio'] = df['Penalty_Risk_Assessment'] / safe_revenue
            else:
                df['Risk_Exposure_Ratio'] = 0.0
        else:
            df['Risk_Exposure_Ratio'] = 0.0
            df['Annual_Revenue'] = 1.0  # Default value to avoid division by zero
        
        # Create Revenue_Category
        if 'Annual_Revenue' in df.columns:
            df['Revenue_Category'] = pd.cut(df['Annual_Revenue'], 
                                           bins=[-1, 10e6, 100e6, 1e9, 10e9, np.inf],
                                           labels=['Startup', 'SME', 'Mid-Market', 'Large', 'Enterprise'])
        else:
            df['Revenue_Category'] = 'Unknown'
        
        # Create ROI_Potential
        if all(col in df.columns for col in ['Penalty_Risk_Assessment', 'Remediation_Cost']):
            safe_remediation = df['Remediation_Cost'].replace(0, 1)
            df['ROI_Potential'] = (df['Penalty_Risk_Assessment'] - df['Remediation_Cost']) / safe_remediation
            df['ROI_Potential'] = df['ROI_Potential'].fillna(0).clip(-10, 10)
        else:
            df['ROI_Potential'] = 0.0
        
        # Ensure all feature columns are present before prediction
        all_features = self.feature_info.get('feature_columns', [])
        for col in all_features:
            if col not in df.columns:
                if col in self.feature_info.get('numerical_columns', []):
                    df[col] = 0.0
                elif col in self.feature_info.get('boolean_columns', []):
                    df[col] = False
                else:
                    df[col] = 'Unknown'
        
        # Return dataframe with columns in the correct order
        return df[all_features]
    
    def predict_scores(self, data):
        """Predict all GRC scores with error handling"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(data)
            predictions = {}
            for score_name, model in self.models.items():
                try:
                    pred = model.predict(processed_data)
                    predictions[score_name] = pred.tolist() if len(pred) > 1 else float(pred[0])
                except Exception as e:
                    logger.error(f"Error predicting {score_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    predictions[score_name] = 0.0 if len(processed_data) == 1 else [0.0] * len(processed_data)
            return predictions
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return {score: 0.0 for score in self.models.keys()}
    
    def get_risk_assessment(self, predictions):
        """Generate comprehensive risk assessment"""
        assessment = {
            'overall_risk_level': 'LOW',
            'priority_actions': []
        }
        # Determine overall risk level
        composite_risk = predictions.get('Composite_Risk_Score', 0)
        if composite_risk > 75:
            assessment['overall_risk_level'] = 'CRITICAL'
        elif composite_risk > 60:
            assessment['overall_risk_level'] = 'HIGH'
        elif composite_risk > 40:
            assessment['overall_risk_level'] = 'MEDIUM'
        
        # Generate specific recommendations
        if predictions.get('Financial_Risk_Score', 0) > 75:
            assessment['priority_actions'].append("Immediate Financial Risk Mitigation required.")
        if predictions.get('Compliance_Score', 100) < 60:
            assessment['priority_actions'].append("Enhance Compliance Controls to address gaps.")
        if predictions.get('Audit_Readiness_Score', 100) < 50:
            assessment['priority_actions'].append("Improve Audit Preparation processes and evidence collection.")
        if predictions.get('Incident_Impact_Score', 0) > 70:
            assessment['priority_actions'].append("Strengthen Incident Response capabilities.")
        
        return assessment

# Example usage and testing
if __name__ == "__main__":
    # This block allows the script to be run directly for testing.
    # It will look for models in the same directory it is located in.
    engine = EnhancedGRCScoreEngine()
    # Sample data for a single prediction
    sample_record = {
        'Applicable_Compliance_Frameworks': 'ISO27001',
        'Control_Category': 'Technical',
        'Control_Status_Distribution': 'Compliant',
        'Compliance_Maturity_Level': 3.5,
        'Control_Testing_Frequency': 'Quarterly',
        'Business_Impact': 'Medium',
        'Annual_Revenue': 50000000.0,
        'Penalty_Risk_Assessment': 250000.0,
        'Remediation_Cost': 50000.0,
        'Asset_Type': 'Database',
        'Data_Sensitivity_Classification': 'Confidential',
        'Geographic_Scope': 'US',
        'Industry_Sector': 'Technology',
        'Audit_Type': 'Internal',
        'Audit_Finding_Severity': 'Medium',
        'Repeat_Finding': False,
        'Evidence_Freshness_Days': 45,
        'Audit_Preparation_Score': 0.75,
        'Incident_Type': 'None',
        'Incident_Notification_Compliance': True,
        'Incident_Cost_Impact': 0.0
    }
    # Predict scores
    predictions = engine.predict_scores(sample_record)
    risk_assessment = engine.get_risk_assessment(predictions)
    print("\n=== Enhanced GRC Scoring Results ===")
    print(f"Compliance Score: {predictions.get('Compliance_Score', 0):.1f}/100")
    print(f"Financial Risk Score: {predictions.get('Financial_Risk_Score', 0):.1f}/100")
    print(f"Asset Risk Index: {predictions.get('Asset_Risk_Index', 0):.1f}/100")
    print(f"Audit Readiness Score: {predictions.get('Audit_Readiness_Score', 0):.1f}/100")
    print(f"Incident Impact Score: {predictions.get('Incident_Impact_Score', 0):.1f}/100")
    print(f"Composite Risk Score: {predictions.get('Composite_Risk_Score', 0):.1f}/100")
    print(f"\nOverall Risk Level: {risk_assessment['overall_risk_level']}")
    if risk_assessment['priority_actions']:
        print("Priority Actions:")
        for action in risk_assessment['priority_actions']:
            print(f"- {action}")
