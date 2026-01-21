import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data(ttl=3600)
def get_occupancy_by_line(df: pd.DataFrame) -> pd.DataFrame:
    occupancy = df.groupby('linea')['ocupacion'].agg(['mean', 'std', 'min', 'max', 'count'])
    occupancy.columns = ['promedio', 'desv_est', 'minimo', 'maximo', 'servicios']
    occupancy = occupancy.reset_index()
    occupancy = occupancy.sort_values('promedio', ascending=False)
    
    return occupancy


@st.cache_data(ttl=3600)
def get_occupancy_by_zone(df: pd.DataFrame) -> pd.DataFrame:
    occupancy = df.groupby('zona')['ocupacion'].agg(['mean', 'std', 'min', 'max', 'count'])
    occupancy.columns = ['promedio', 'desv_est', 'minimo', 'maximo', 'servicios']
    occupancy = occupancy.reset_index()
    occupancy = occupancy.sort_values('promedio', ascending=False)
    
    return occupancy


@st.cache_data(ttl=3600)
def get_occupancy_by_day(df: pd.DataFrame) -> pd.DataFrame:
    occupancy = df.groupby('nombre_dia')['ocupacion'].agg(['mean', 'std'])
    occupancy.columns = ['promedio', 'desv_est']
    occupancy = occupancy.reset_index()
    
    # Order by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    occupancy['nombre_dia'] = pd.Categorical(
        occupancy['nombre_dia'],
        categories=day_order,
        ordered=True
    )
    occupancy = occupancy.sort_values('nombre_dia')
    
    return occupancy


@st.cache_data(ttl=3600)
def get_occupancy_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    heatmap = df.pivot_table(
        values='ocupacion',
        index='hora',
        columns='dia_semana',
        aggfunc='mean'
    )
    
    # Rename columns to day names
    day_names = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
                 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    heatmap.columns = [day_names.get(col, col) for col in heatmap.columns]
    
    return heatmap


@st.cache_data(ttl=3600)
def get_occupancy_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    occupancy = df.groupby('hora')['ocupacion'].agg(['mean', 'std', 'min', 'max'])
    occupancy.columns = ['promedio', 'desv_est', 'minimo', 'maximo']
    occupancy = occupancy.reset_index()
    occupancy = occupancy.sort_values('hora')
    
    return occupancy


@st.cache_data(ttl=3600)
def get_critical_lines(df: pd.DataFrame, threshold_occupancy: float = 75) -> pd.DataFrame:
    line_stats = df.groupby('linea').agg({
        'ocupacion': 'mean',
        'retraso': 'mean',
        'pasajeros': 'sum'
    })
    
    line_stats.columns = ['ocupacion_promedio', 'retraso_promedio', 'pasajeros_total']
    
    # Critical lines: above threshold occupancy and above median delay
    median_delay = df['retraso'].median()
    
    # Fill NaN values with False to avoid ambiguous comparison
    mask1 = (line_stats['ocupacion_promedio'] >= threshold_occupancy).fillna(False)
    mask2 = (line_stats['retraso_promedio'] > median_delay).fillna(False)
    
    critical = line_stats[mask1 & mask2]
    
    critical = critical.reset_index()
    critical = critical.sort_values('ocupacion_promedio', ascending=False)
    
    return critical


@st.cache_data(ttl=3600)
def get_occupancy_by_time_slot(df: pd.DataFrame) -> pd.DataFrame:
    occupancy = df.groupby('franja_horaria')['ocupacion'].agg(['mean', 'std', 'min', 'max'])
    occupancy.columns = ['promedio', 'desv_est', 'minimo', 'maximo']
    occupancy = occupancy.reset_index()
    
    # Order time slots
    slot_order = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
    occupancy['franja_horaria'] = pd.Categorical(
        occupancy['franja_horaria'],
        categories=slot_order,
        ordered=True
    )
    occupancy = occupancy.sort_values('franja_horaria')
    
    return occupancy
