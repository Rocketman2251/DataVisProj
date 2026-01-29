"""
Anomaly Classifier Module
Classifies detected anomalies into specific types
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Optional, Tuple
import os


class AnomalyClassifier:
    """
    Classifies anomalies into specific types:
    - LUNCH_BREAK: Driver lunch hours
    - SPECIAL_EVENT: Concert, match, demonstration
    - WEATHER: Adverse weather conditions
    - ACCIDENT: Traffic accident
    - STRIKE: Labor strike
    - NORMAL: Normal operation
    """
    
    ANOMALY_TYPES = [
        'LUNCH_BREAK',
        'SPECIAL_EVENT', 
        'WEATHER',
        'ACCIDENT',
        'STRIKE',
        'NORMAL'
    ]
    
    def __init__(self, random_state: int = 42):
        """
        Initialize classifier
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """
        Train the classifier
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (n_samples,)
            feature_names: List of feature names (optional)
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train model
        self.model.fit(X, y_encoded)
        
        self.is_fitted = True
        self.feature_names = feature_names
        
        print(f"✓ Clasificador entrenado con {X.shape[0]} muestras")
        print(f"  Clases: {list(self.label_encoder.classes_)}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly types
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        y_pred_encoded = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def get_predictions_with_confidence(self, X: np.ndarray) -> pd.DataFrame:
        """
        Get predictions with confidence scores
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with predictions and confidence
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Get confidence (max probability)
        confidence = np.max(probabilities, axis=1) * 100
        
        result = pd.DataFrame({
            'tipo_anomalia': predictions,
            'confianza': confidence
        })
        
        # Add probability for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            result[f'prob_{class_name}'] = probabilities[:, i] * 100
        
        return result
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate classifier performance
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        # Classification report
        report = classification_report(
            y_test_encoded, 
            y_pred_encoded,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        return {
            'accuracy': report['accuracy'],
            'report': report,
            'confusion_matrix': cm,
            'classes': list(self.label_encoder.classes_)
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with CV scores
        """
        y_encoded = self.label_encoder.fit_transform(y)
        
        scores = cross_val_score(
            self.model, X, y_encoded, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        return {
            'cv_scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
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
            'label_encoder': self.label_encoder,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"✓ Clasificador guardado en: {filepath}")
        
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
        self.label_encoder = model_data['label_encoder']
        self.random_state = model_data['random_state']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        
        print(f"✓ Clasificador cargado desde: {filepath}")


class LabelingAssistant:
    """
    Helps with semi-automated labeling of anomalies
    """
    
    @staticmethod
    def suggest_labels(df: pd.DataFrame) -> pd.Series:
        """
        Suggest labels based on heuristic rules
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with suggested labels
        """
        labels = []
        
        for _, row in df.iterrows():
            # Rule-based suggestions
            if row['es_hora_almuerzo'] == 1 and row['pasajeros'] < row.get('pasajeros_esperado', 0) * 0.7:
                labels.append('LUNCH_BREAK')
            elif row['retraso'] > 20:
                labels.append('ACCIDENT')
            elif row['ocupacion'] < 30 and row['es_hora_pico'] == 1:
                labels.append('WEATHER')
            elif row['ocupacion'] > 95:
                labels.append('SPECIAL_EVENT')
            elif row.get('variacion_dia_anterior', 0) < -0.5:
                labels.append('STRIKE')
            else:
                labels.append('NORMAL')
        
        return pd.Series(labels, index=df.index)
    
    @staticmethod
    def create_labeling_template(df_anomalies: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
        """
        Create a template for manual labeling
        
        Args:
            df_anomalies: DataFrame with detected anomalies
            n_samples: Number of samples to include
            
        Returns:
            DataFrame ready for manual labeling
        """
        # Sample diverse anomalies
        if len(df_anomalies) > n_samples:
            sample = df_anomalies.sample(n=n_samples, random_state=42)
        else:
            sample = df_anomalies
        
        # Create template with key columns
        template = sample[[
            'fecha', 'hora', 'linea', 'zona', 
            'pasajeros', 'ocupacion', 'retraso',
            'anomalia_confianza'
        ]].copy()
        
        # Add suggested label
        template['tipo_sugerido'] = LabelingAssistant.suggest_labels(sample)
        template['tipo_confirmado'] = ''  # To be filled manually
        template['notas'] = ''  # For additional comments
        
        return template


if __name__ == "__main__":
    print("Anomaly Classification Module")
    print("=" * 60)
    print("\nThis module classifies detected anomalies into specific types.")
    print("\nSupported anomaly types:")
    for i, anomaly_type in enumerate(AnomalyClassifier.ANOMALY_TYPES, 1):
        print(f"  {i}. {anomaly_type}")
    print("\nKey features:")
    print("  - Random Forest classification")
    print("  - Class balancing for imbalanced data")
    print("  - Prediction confidence scores")
    print("  - Feature importance analysis")
    print("  - Cross-validation support")
    print("\nExample usage:")
    print("  classifier = AnomalyClassifier()")
    print("  classifier.fit(X_train, y_train)")
    print("  predictions = classifier.predict(X_test)")
