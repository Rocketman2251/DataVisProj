"""
Anomaly Detection Module using Isolation Forest
Detects unusual patterns in transport data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Optional, Tuple
import os


class TransportAnomalyDetector:
    """
    Detects anomalies in transport data using Isolation Forest
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None):
        """
        Train the anomaly detection model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names (optional)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        self.feature_names = feature_names
        
        print(f"✓ Modelo entrenado con {X.shape[0]} muestras y {X.shape[1]} características")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of predictions: 1 = normal, -1 = anomaly
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        
        return scores
    
    def get_anomaly_info(self, df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
        """
        Get detailed information about detected anomalies
        
        Args:
            df: Original DataFrame
            X: Feature matrix
            
        Returns:
            DataFrame with anomaly information
        """
        predictions = self.predict(X)
        scores = self.predict_proba(X)
        
        result = df.copy()
        result['es_anomalia'] = (predictions == -1).astype(int)
        result['anomalia_score'] = scores
        
        # Normalize scores to 0-100 (0 = most anomalous)
        min_score, max_score = scores.min(), scores.max()
        result['anomalia_confianza'] = 100 * (1 - (scores - min_score) / (max_score - min_score + 1e-6))
        
        return result
    
    def get_statistics(self, predictions: np.ndarray) -> dict:
        """
        Get statistics about anomaly detection
        
        Args:
            predictions: Array of predictions
            
        Returns:
            Dictionary with statistics
        """
        n_total = len(predictions)
        n_anomalies = np.sum(predictions == -1)
        n_normal = np.sum(predictions == 1)
        
        return {
            'total_samples': n_total,
            'anomalies': n_anomalies,
            'normal': n_normal,
            'anomaly_rate': n_anomalies / n_total * 100,
            'contamination_rate': self.contamination * 100
        }
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"✓ Modelo guardado en: {filepath}")
        
    def load_model(self, filepath: str):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to load model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        
        print(f"✓ Modelo cargado desde: {filepath}")


class AnomalyAnalyzer:
    """
    Analyzes detected anomalies to provide insights
    """
    
    @staticmethod
    def analyze_temporal_patterns(df_anomalies: pd.DataFrame) -> dict:
        """
        Analyze temporal patterns in anomalies
        
        Args:
            df_anomalies: DataFrame with detected anomalies
            
        Returns:
            Dictionary with temporal analysis
        """
        anomalies = df_anomalies[df_anomalies['es_anomalia'] == 1]
        
        if len(anomalies) == 0:
            return {'message': 'No anomalies detected'}
        
        return {
            'by_hour': anomalies['hora'].value_counts().to_dict(),
            'by_day': anomalies['dia_semana'].value_counts().to_dict(),
            'by_line': anomalies['linea'].value_counts().to_dict(),
            'by_zone': anomalies['zona'].value_counts().to_dict(),
            'lunch_hours': len(anomalies[anomalies['es_hora_almuerzo'] == 1]),
            'peak_hours': len(anomalies[anomalies['es_hora_pico'] == 1]),
        }
    
    @staticmethod
    def get_top_anomalies(df_anomalies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Get the most anomalous records
        
        Args:
            df_anomalies: DataFrame with anomaly scores
            n: Number of top anomalies to return
            
        Returns:
            DataFrame with top n anomalies
        """
        anomalies = df_anomalies[df_anomalies['es_anomalia'] == 1]
        return anomalies.nlargest(n, 'anomalia_confianza')
    
    @staticmethod
    def suggest_anomaly_types(df_anomalies: pd.DataFrame) -> pd.DataFrame:
        """
        Suggest potential anomaly types based on patterns
        
        Args:
            df_anomalies: DataFrame with detected anomalies
            
        Returns:
            DataFrame with suggested anomaly types
        """
        anomalies = df_anomalies[df_anomalies['es_anomalia'] == 1].copy()
        
        if len(anomalies) == 0:
            return pd.DataFrame()
        
        # Simple rule-based suggestions (these will be refined by the classifier)
        conditions = [
            (anomalies['es_hora_almuerzo'] == 1) & (anomalies['pasajeros'] < anomalies['pasajeros_esperado']),
            (anomalies['retraso'] > 15),
            (anomalies['ocupacion'] < 30) & (anomalies['es_hora_pico'] == 1),
            (anomalies['ocupacion'] > 90),
        ]
        
        choices = [
            'LUNCH_BREAK',
            'DELAY_ACCIDENT',
            'LOW_DEMAND',
            'HIGH_DEMAND'
        ]
        
        anomalies['tipo_sugerido'] = np.select(conditions, choices, default='UNKNOWN')
        
        return anomalies


if __name__ == "__main__":
    print("Transport Anomaly Detection Module")
    print("=" * 60)
    print("\nThis module uses Isolation Forest to detect anomalies")
    print("in transport data.")
    print("\nKey features:")
    print("  - Unsupervised anomaly detection")
    print("  - Anomaly scoring and confidence")
    print("  - Temporal pattern analysis")
    print("  - Model persistence (save/load)")
    print("\nExample usage:")
    print("  detector = TransportAnomalyDetector(contamination=0.1)")
    print("  detector.fit(X_train)")
    print("  predictions = detector.predict(X_test)")
