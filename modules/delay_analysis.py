import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict


@st.cache_data(ttl=3600)
def get_delay_statistics(df: pd.DataFrame) -> Dict:
    try:
        if 'retraso' not in df.columns or len(df) == 0:
            return {
                'promedio': 0, 'mediana': 0, 'desv_est': 0,
                'minimo': 0, 'maximo': 0, 'q25': 0, 'q75': 0
            }
        
        return {
            'promedio': df['retraso'].mean(),
            'mediana': df['retraso'].median(),
            'desv_est': df['retraso'].std(),
            'minimo': df['retraso'].min(),
            'maximo': df['retraso'].max(),
            'q25': df['retraso'].quantile(0.25),
            'q75': df['retraso'].quantile(0.75)
        }
    except Exception:
        return {
            'promedio': 0, 'mediana': 0, 'desv_est': 0,
            'minimo': 0, 'maximo': 0, 'q25': 0, 'q75': 0
        }


@st.cache_data(ttl=3600)
def get_delay_histogram(df: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    hist, bin_edges = np.histogram(df['retraso'], bins=bins)
    
    histogram_df = pd.DataFrame({
        'rango': [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(hist))],
        'frecuencia': hist
    })
    
    return histogram_df


@st.cache_data(ttl=3600)
def get_delay_by_line(df: pd.DataFrame) -> pd.DataFrame:
    delay = df.groupby('linea')['retraso'].agg(['mean', 'std', 'min', 'max', 'count'])
    delay.columns = ['promedio', 'desv_est', 'minimo', 'maximo', 'servicios']
    delay = delay.reset_index()
    delay = delay.sort_values('promedio', ascending=False)
    
    return delay


@st.cache_data(ttl=3600)
def get_delay_by_zone(df: pd.DataFrame) -> pd.DataFrame:
    delay = df.groupby('zona')['retraso'].agg(['mean', 'std', 'min', 'max', 'count'])
    delay.columns = ['promedio', 'desv_est', 'minimo', 'maximo', 'servicios']
    delay = delay.reset_index()
    delay = delay.sort_values('promedio', ascending=False)
    
    return delay


@st.cache_data(ttl=3600)
def get_delay_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    delay = df.groupby('hora')['retraso'].agg(['mean', 'std', 'min', 'max'])
    delay.columns = ['promedio', 'desv_est', 'minimo', 'maximo']
    delay = delay.reset_index()
    delay = delay.sort_values('hora')
    
    return delay


@st.cache_data(ttl=3600)
def get_punctuality_rate(df: pd.DataFrame, threshold: int = 5) -> dict:
    on_time = (df['retraso'] < threshold).sum()
    total = len(df)
    rate = (on_time / total * 100) if total > 0 else 0
    
    return {
        'puntual': on_time,
        'total': total,
        'porcentaje': rate,
        'retrasada': total - on_time
    }


@st.cache_data(ttl=3600)
def get_delay_by_day(df: pd.DataFrame) -> pd.DataFrame:
    delay = df.groupby('nombre_dia')['retraso'].agg(['mean', 'std', 'count'])
    delay.columns = ['promedio', 'desv_est', 'servicios']
    delay = delay.reset_index()
    
    # Order by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    delay['nombre_dia'] = pd.Categorical(
        delay['nombre_dia'],
        categories=day_order,
        ordered=True
    )
    delay = delay.sort_values('nombre_dia')
    
    return delay


@st.cache_data(ttl=3600)
def get_delay_by_time_slot(df: pd.DataFrame) -> pd.DataFrame:
    delay = df.groupby('franja_horaria')['retraso'].agg(['mean', 'std', 'count'])
    delay.columns = ['promedio', 'desv_est', 'servicios']
    delay = delay.reset_index()
    
    # Order time slots
    slot_order = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
    delay['franja_horaria'] = pd.Categorical(
        delay['franja_horaria'],
        categories=slot_order,
        ordered=True
    )
    delay = delay.sort_values('franja_horaria')
    
    return delay


@st.cache_data(ttl=3600)
def get_punctuality_by_line(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    line_data = df.groupby('linea').agg({
        'es_puntual': 'sum',
        'linea': 'count'
    })
    
    line_data.columns = ['puntual', 'total']
    line_data['porcentaje'] = (line_data['puntual'] / line_data['total'] * 100).round(2)
    line_data = line_data.reset_index()
    line_data = line_data.sort_values('porcentaje', ascending=False)
    
    return line_data
