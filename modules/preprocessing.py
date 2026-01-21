import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


@st.cache_data(ttl=3600, show_spinner=False)
def clean_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df = df.copy()
    
    initial_rows = len(df)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values in critical columns
    for col in ['fecha', 'hora', 'linea', 'zona']:
        if col in df.columns:
            df = df.dropna(subset=[col])
    
    # Handle missing values in numeric columns with median imputation
    numeric_cols = ['pasajeros', 'ocupacion', 'retraso']
    for col in numeric_cols:
        if col in df.columns:
            # Calculate median for imputation
            median_val = pd.to_numeric(df[col], errors='coerce').median()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(median_val, inplace=True)
    
    # Remove rows with still-missing critical numeric values
    df = df.dropna(subset=numeric_cols)
    
    # Convert and clean numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with negative values in numeric columns
    df = df[(df['pasajeros'] >= 0) & (df['ocupacion'] >= 0) & (df['retraso'] >= 0)]
    
    # Clamp occupancy between 0 and 100
    if 'ocupacion' in df.columns:
        df['ocupacion'] = df['ocupacion'].clip(0, 100)
    
    # Remove obvious outliers (using IQR method)
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    if verbose:
        print(f"Rows removed: {initial_rows - len(df)}")
    
    return df


def transform_datetime_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df = df.copy()
    
    # Parse fecha with multiple formats
    if 'fecha' in df.columns:
        # Try to convert to datetime
        df['fecha'] = pd.to_datetime(
            df['fecha'],
            format='mixed',
            errors='coerce',
            dayfirst=True
        )
        
        # Count and remove invalid dates
        invalid_dates = df['fecha'].isnull().sum()
        if invalid_dates > 0:
            # Replace NaT with sequential dates starting from a default
            nat_mask = df['fecha'].isnull()
            if nat_mask.any():
                min_date = df.loc[~nat_mask, 'fecha'].min()
                if pd.isna(min_date):
                    min_date = pd.Timestamp('2024-01-01')
                # Replace NaT with sequential dates
                nat_indices = df[nat_mask].index
                df.loc[nat_indices, 'fecha'] = pd.date_range(
                    start=min_date,
                    periods=len(nat_indices),
                    freq='D'
                )
            
            if verbose:
                print(f"Invalid dates replaced: {invalid_dates}")
        
        # Ensure no NaT values remain
        df = df.dropna(subset=['fecha'])
    
    # Parse hora (must be 0-23)
    if 'hora' in df.columns:
        df['hora'] = pd.to_numeric(df['hora'], errors='coerce')
        # Replace NaN with median hour
        if df['hora'].isnull().any():
            median_hora = df['hora'].median()
            if pd.isna(median_hora):
                median_hora = 12
            df['hora'] = df['hora'].fillna(median_hora)
        df['hora'] = df['hora'].astype('int64').clip(0, 23)
    
    # Create day of week (0=Monday, 6=Sunday)
    if 'fecha' in df.columns:
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['nombre_dia'] = df['fecha'].dt.day_name()
    
    # Create time slot (franja horaria)
    if 'hora' in df.columns:
        df['franja_horaria'] = pd.cut(
            df['hora'],
            bins=[-1, 6, 12, 18, 24],
            labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'],
            right=False
        )
    
    # Create weekend indicator
    if 'dia_semana' in df.columns:
        df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
        df['tipo_dia'] = df['es_fin_semana'].apply(
            lambda x: 'Fin de semana' if x else 'Entre semana'
        )
    
    # Create date components
    if 'fecha' in df.columns:
        df['año'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month
        df['semana'] = df['fecha'].dt.isocalendar().week
        df['dia_mes'] = df['fecha'].dt.day
    
    return df


def create_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Calculate if service is on time (delay < 5 minutes)
    if 'retraso' in df.columns:
        df['es_puntual'] = (df['retraso'] < 5).astype(int)
    
    # Categorize delay severity
    if 'retraso' in df.columns:
        df['severidad_retraso'] = pd.cut(
            df['retraso'],
            bins=[-0.1, 5, 15, 30, float('inf')],
            labels=['Puntual', 'Leve', 'Moderado', 'Severo'],
            right=False
        )
    
    # Categorize occupancy level
    if 'ocupacion' in df.columns:
        df['nivel_ocupacion'] = pd.cut(
            df['ocupacion'],
            bins=[-0.1, 25, 50, 75, 100.1],
            labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'],
            right=False
        )
    
    # Create occupancy-delay risk score
    if 'ocupacion' in df.columns and 'retraso' in df.columns:
        occ_norm = (df['ocupacion'] - df['ocupacion'].min()) / (df['ocupacion'].max() - df['ocupacion'].min() + 1e-8)
        delay_norm = (df['retraso'] - df['retraso'].min()) / (df['retraso'].max() - df['retraso'].min() + 1e-8)
        df['riesgo_operacional'] = (occ_norm * 0.6 + delay_norm * 0.4) * 100
    
    return df


def validate_processed_data(df: pd.DataFrame) -> Tuple[bool, list]:
    issues = []
    
    required_cols = ['fecha', 'hora', 'linea', 'zona', 'pasajeros', 'ocupacion', 'retraso']
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Columna requerida faltante: {col}")
    
    if len(df) == 0:
        issues.append("El dataframe está vacío después del procesamiento")
    
    numeric_cols = ['hora', 'pasajeros', 'ocupacion', 'retraso']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype not in ['int64', 'float64', 'Int64']:
                issues.append(f"Columna {col} no tiene tipo numérico")
    
    return len(issues) == 0, issues


@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_full_pipeline(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f"Inicio: {len(df)} filas")
    
    df = clean_data(df, verbose=verbose)
    if verbose:
        print(f"Después de limpieza: {len(df)} filas")
    
    df = transform_datetime_features(df, verbose=verbose)
    if verbose:
        print(f"Después de transformación: {len(df)} filas")
    
    df = create_derived_metrics(df)
    if verbose:
        print(f"Derivadas creadas: {len(df)} filas")
    
    is_valid, issues = validate_processed_data(df)
    if not is_valid and verbose:
        for issue in issues:
            print(f"⚠️ Advertencia: {issue}")
    
    return df
