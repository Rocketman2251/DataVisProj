"""
Feature Engineering Module for Transport Anomaly Detection
Extracts and creates features from raw transport data for ML models
"""

import numpy as np
import pandas as pd
from typing import Tuple


class TransportFeatureEngineer:
    """
    Creates features from transport data for anomaly detection
    """
    
    def __init__(self):
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical temporal features
        
        Args:
            df: DataFrame with 'hora' and 'dia_semana' columns
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        
        # Cyclical hour encoding (0-23)
        df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
        
        # Cyclical day encoding (0-6)
        df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        
        # Boolean features
        df['es_hora_pico'] = df['hora'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df['es_hora_almuerzo'] = df['hora'].isin([12, 13, 14]).astype(int)
        df['es_madrugada'] = df['hora'].isin([0, 1, 2, 3, 4, 5]).astype(int)
        df['es_noche'] = df['hora'].isin([20, 21, 22, 23]).astype(int)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features based on historical patterns
        
        Args:
            df: DataFrame with transport data
            
        Returns:
            DataFrame with added statistical features
        """
        df = df.copy()
        
        # Moving averages for passengers
        df = df.sort_values(['linea', 'fecha', 'hora'])
        df['pasajeros_ma3'] = df.groupby('linea')['pasajeros'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Standard deviation of occupancy
        df['ocupacion_std'] = df.groupby(['linea', 'hora'])['ocupacion'].transform('std')
        
        # Delay percentile within line
        df['retraso_percentil'] = df.groupby('linea')['retraso'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # Expected occupancy for this hour/line
        df['ocupacion_esperada'] = df.groupby(['linea', 'hora'])['ocupacion'].transform('mean')
        df['ratio_ocupacion'] = df['ocupacion'] / (df['ocupacion_esperada'] + 1e-6)
        
        # Expected passengers for this hour/line
        df['pasajeros_esperado'] = df.groupby(['linea', 'hora'])['pasajeros'].transform('mean')
        df['ratio_pasajeros'] = df['pasajeros'] / (df['pasajeros_esperado'] + 1e-6)
        
        return df
    
    def create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on broader context (zone, day, etc.)
        
        Args:
            df: DataFrame with transport data
            
        Returns:
            DataFrame with added contextual features
        """
        df = df.copy()
        
        # Total passengers in zone at this hour
        df['pasajeros_zona_total'] = df.groupby(['zona', 'fecha', 'hora'])['pasajeros'].transform('sum')
        
        # Number of active lines in zone
        df['lineas_activas_zona'] = df.groupby(['zona', 'fecha', 'hora'])['linea'].transform('nunique')
        
        # Average occupancy in zone
        df['ocupacion_promedio_zona'] = df.groupby(['zona', 'fecha', 'hora'])['ocupacion'].transform('mean')
        
        # Variation compared to previous day
        df = df.sort_values(['linea', 'fecha', 'hora'])
        df['pasajeros_dia_anterior'] = df.groupby(['linea', 'hora'])['pasajeros'].shift(1)
        df['variacion_dia_anterior'] = (
            (df['pasajeros'] - df['pasajeros_dia_anterior']) / 
            (df['pasajeros_dia_anterior'] + 1e-6)
        )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features combining multiple variables
        
        Args:
            df: DataFrame with transport data
            
        Returns:
            DataFrame with added interaction features
        """
        df = df.copy()
        
        # Occupancy-delay interaction
        df['ocupacion_x_retraso'] = df['ocupacion'] * df['retraso']
        
        # Passengers-occupancy ratio
        df['pasajeros_por_ocupacion'] = df['pasajeros'] / (df['ocupacion'] + 1e-6)
        
        # Peak hour with high occupancy flag
        df['pico_alta_ocupacion'] = (
            (df['es_hora_pico'] == 1) & (df['ocupacion'] > 70)
        ).astype(int)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Apply all feature engineering steps
        
        Args:
            df: Raw transport data DataFrame
            
        Returns:
            Tuple of (DataFrame with features, list of feature column names)
        """
        # Ensure required columns exist
        required_cols = ['fecha', 'hora', 'linea', 'zona', 'pasajeros', 'ocupacion', 'retraso']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Apply feature engineering
        df = self.create_temporal_features(df)
        df = self.create_statistical_features(df)
        df = self.create_contextual_features(df)
        df = self.create_interaction_features(df)
        
        # Fill any NaN values created during feature engineering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Define feature columns for ML
        self.feature_names = [
            # Temporal
            'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos',
            'es_hora_pico', 'es_hora_almuerzo', 'es_madrugada', 'es_noche',
            # Statistical
            'pasajeros_ma3', 'ocupacion_std', 'retraso_percentil',
            'ratio_ocupacion', 'ratio_pasajeros',
            # Contextual
            'pasajeros_zona_total', 'lineas_activas_zona', 
            'ocupacion_promedio_zona', 'variacion_dia_anterior',
            # Interaction
            'ocupacion_x_retraso', 'pasajeros_por_ocupacion', 'pico_alta_ocupacion',
            # Original features
            'pasajeros', 'ocupacion', 'retraso'
        ]
        
        return df, self.feature_names
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature matrix for ML models
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Numpy array of features
        """
        if not self.feature_names:
            raise ValueError("Must call engineer_features() first")
        
        return df[self.feature_names].values


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module for Transport Anomaly Detection")
    print("=" * 60)
    print("\nThis module creates features for ML anomaly detection.")
    print("\nExample features created:")
    print("  - Temporal: hora_sin, hora_cos, es_hora_pico")
    print("  - Statistical: pasajeros_ma3, ocupacion_std")
    print("  - Contextual: pasajeros_zona_total, lineas_activas_zona")
    print("  - Interaction: ocupacion_x_retraso, pico_alta_ocupacion")
    print("\nUse TransportFeatureEngineer class to engineer features from raw data.")
