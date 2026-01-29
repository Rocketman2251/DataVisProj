"""
Training Script for Transport Anomaly Detection System
Trains both the anomaly detector and classifier models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# Import custom modules
from modules.ml_feature_engineering import TransportFeatureEngineer
from modules.ml_anomaly_detection import TransportAnomalyDetector
from modules.ml_classifier import AnomalyClassifier, LabelingAssistant


def load_data(filepath: str) -> pd.DataFrame:
    """Load and basic preprocessing of transport data"""
    print(f" Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert date to datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Add day of week if not present
    if 'dia_semana' not in df.columns:
        df['dia_semana'] = df['fecha'].dt.dayofweek
    
    print(f"✓ Datos cargados: {len(df)} registros")
    return df


def train_anomaly_detector(df: pd.DataFrame, contamination: float = 0.1) -> tuple:
    """
    Train the anomaly detection model
    
    Returns:
        Tuple of (detector, df_with_anomalies, feature_matrix)
    """
    print("\n" + "="*60)
    print("PASO 1: ENTRENAMIENTO DEL DETECTOR DE ANOMALÍAS")
    print("="*60)
    
    # Feature engineering
    print("\n1.1 Ingeniería de características...")
    engineer = TransportFeatureEngineer()
    df_features, feature_names = engineer.engineer_features(df)
    print(f"✓ {len(feature_names)} características creadas")
    
    # Get feature matrix
    X = engineer.get_feature_matrix(df_features)
    print(f"✓ Matriz de características: {X.shape}")
    
    # Train detector
    print(f"\n1.2 Entrenando Isolation Forest (contamination={contamination})...")
    detector = TransportAnomalyDetector(contamination=contamination)
    detector.fit(X, feature_names=feature_names)
    
    # Get anomaly predictions
    print("\n1.3 Detectando anomalías...")
    df_with_anomalies = detector.get_anomaly_info(df_features, X)
    
    stats = detector.get_statistics(detector.predict(X))
    print(f"\n Estadísticas de detección:")
    print(f"   Total de muestras: {stats['total_samples']:,}")
    print(f"   Anomalías detectadas: {stats['anomalies']:,} ({stats['anomaly_rate']:.2f}%)")
    print(f"   Normales: {stats['normal']:,}")
    
    return detector, df_with_anomalies, X, feature_names


def create_training_labels(df_anomalies: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    """
    Create labeled dataset for classifier training
    
    In a real scenario, you would:
    1. Export this to CSV
    2. Manually review and correct labels
    3. Import back for training
    
    For this demo, we use heuristic rules
    """
    print("\n" + "="*60)
    print("PASO 2: CREACIÓN DE ETIQUETAS PARA CLASIFICACIÓN")
    print("="*60)
    
    # Filter to anomalies only
    anomalies_only = df_anomalies[df_anomalies['es_anomalia'] == 1].copy()
    
    if len(anomalies_only) == 0:
        raise ValueError("No anomalies detected! Adjust contamination parameter.")
    
    print(f"\n2.1 Anomalías detectadas: {len(anomalies_only)}")
    
    # Sample for labeling
    if len(anomalies_only) > n_samples:
        labeled_sample = anomalies_only.sample(n=n_samples, random_state=42)
        print(f"✓ Muestreadas {n_samples} anomalías para etiquetado")
    else:
        labeled_sample = anomalies_only
        print(f"✓ Usando todas las {len(labeled_sample)} anomalías")
    
    # Generate labels using heuristic rules
    print("\n2.2 Generando etiquetas heurísticas...")
    labels = []
    
    for _, row in labeled_sample.iterrows():
        # Lunch break detection
        if (row['es_hora_almuerzo'] == 1 and 
            row['pasajeros'] < row.get('pasajeros_esperado', row['pasajeros']) * 0.7):
            labels.append('LUNCH_BREAK')
        
        # Accident detection (high delays)
        elif row['retraso'] > 15:
            labels.append('ACCIDENT')
        
        # Weather detection (low occupancy during peak hours)
        elif row['ocupacion'] < 40 and row['es_hora_pico'] == 1:
            labels.append('WEATHER')
        
        # Special event (very high occupancy)
        elif row['ocupacion'] > 90:
            labels.append('SPECIAL_EVENT')
        
        # Strike (large drop vs previous day)
        elif row.get('variacion_dia_anterior', 0) < -0.4:
            labels.append('STRIKE')
        
        else:
            labels.append('NORMAL')
    
    labeled_sample['tipo_anomalia'] = labels
    
    # Print distribution
    print("\n Distribución de etiquetas:")
    for label, count in labeled_sample['tipo_anomalia'].value_counts().items():
        print(f"   {label}: {count} ({count/len(labeled_sample)*100:.1f}%)")
    
    return labeled_sample


def train_classifier(df_labeled: pd.DataFrame, feature_names: list) -> AnomalyClassifier:
    """Train the anomaly classifier"""
    print("\n" + "="*60)
    print("PASO 3: ENTRENAMIENTO DEL CLASIFICADOR")
    print("="*60)
    
    # Prepare features and labels
    X = df_labeled[feature_names].values
    y = df_labeled['tipo_anomalia'].values
    
    print(f"\n3.1 Preparando datos de entrenamiento...")
    print(f"   Muestras: {len(X)}")
    print(f"   Características: {len(feature_names)}")
    print(f"   Clases: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n   Training set: {len(X_train)} muestras")
    print(f"   Test set: {len(X_test)} muestras")
    
    # Train classifier
    print(f"\n3.2 Entrenando Random Forest Classifier...")
    classifier = AnomalyClassifier(random_state=42)
    classifier.fit(X_train, y_train, feature_names=feature_names)
    
    # Evaluate
    print(f"\n3.3 Evaluando modelo...")
    evaluation = classifier.evaluate(X_test, y_test)
    
    print(f"\n Métricas de rendimiento:")
    print(f"   Accuracy: {evaluation['accuracy']:.3f}")
    print(f"\n   Por clase:")
    for class_name in evaluation['classes']:
        metrics = evaluation['report'][class_name]
        print(f"   {class_name}:")
        print(f"      Precision: {metrics['precision']:.3f}")
        print(f"      Recall: {metrics['recall']:.3f}")
        print(f"      F1-Score: {metrics['f1-score']:.3f}")
    
    # Feature importance
    print(f"\n3.4 Características más importantes:")
    importance = classifier.get_feature_importance(top_n=5)
    for idx, row in importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return classifier


def save_models(detector: TransportAnomalyDetector, 
                classifier: AnomalyClassifier,
                output_dir: str = "models"):
    """Save trained models"""
    print("\n" + "="*60)
    print("PASO 4: GUARDANDO MODELOS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detector
    detector_path = os.path.join(output_dir, "anomaly_detector.pkl")
    detector.save_model(detector_path)
    
    # Save classifier
    classifier_path = os.path.join(output_dir, "anomaly_classifier.pkl")
    classifier.save_model(classifier_path)
    
    print(f"\n✓ Modelos guardados en: {output_dir}/")
    print(f"   - anomaly_detector.pkl")
    print(f"   - anomaly_classifier.pkl")


def main():
    """Main training pipeline"""
    print("="*60)
    print("SISTEMA DE DETECCIÓN DE ANOMALÍAS EN TRANSPORTE URBANO")
    print("Pipeline de Entrenamiento")
    print("="*60)
    
    # Configuration
    DATA_FILE = "sample_data_completo.csv"  # Your data file
    CONTAMINATION = 0.10  # Expected anomaly rate
    N_LABELS = 500  # Number of samples to label
    OUTPUT_DIR = "models"
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"\n Error: No se encontró el archivo {DATA_FILE}")
        print("   Asegúrate de tener los datos en el directorio actual.")
        sys.exit(1)
    
    try:
        # Load data
        df = load_data(DATA_FILE)
        
        # Train anomaly detector
        detector, df_anomalies, X, feature_names = train_anomaly_detector(
            df, contamination=CONTAMINATION
        )
        
        # Create labeled dataset
        df_labeled = create_training_labels(df_anomalies, n_samples=N_LABELS)
        
        # Train classifier
        classifier = train_classifier(df_labeled, feature_names)
        
        # Save models
        save_models(detector, classifier, output_dir=OUTPUT_DIR)
        
        print("\n" + "="*60)
        print(" ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\nPróximos pasos:")
        print("1. Revisar y ajustar las etiquetas manualmente si es necesario")
        print("2. Re-entrenar el clasificador con etiquetas corregidas")
        print("3. Integrar los modelos en la aplicación Streamlit")
        print("4. Evaluar el rendimiento en datos nuevos")
        
    except Exception as e:
        print(f"\n Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
