import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional


def safe_groupby_aggregate(df: pd.DataFrame, groupby_col: str, agg_col: str) -> Optional[pd.DataFrame]:
    try:
        if groupby_col not in df.columns or agg_col not in df.columns:
            return None
        
        result = df.groupby(groupby_col)[agg_col].agg(['sum', 'mean', 'count']).reset_index()
        result.columns = [groupby_col, 'total', 'promedio', 'servicios']
        return result
    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def get_hourly_passengers(df: pd.DataFrame) -> pd.DataFrame:
    if 'hora' not in df.columns or 'pasajeros' not in df.columns:
        return pd.DataFrame()
    
    hourly = safe_groupby_aggregate(df, 'hora', 'pasajeros')
    if hourly is not None:
        hourly = hourly.sort_values('hora')
        return hourly
    
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_daily_passengers(df: pd.DataFrame) -> pd.DataFrame:
    if 'nombre_dia' not in df.columns or 'pasajeros' not in df.columns:
        return pd.DataFrame()
    
    daily = safe_groupby_aggregate(df, 'nombre_dia', 'pasajeros')
    if daily is not None:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily['nombre_dia'] = pd.Categorical(
            daily['nombre_dia'],
            categories=day_order,
            ordered=True
        )
        daily = daily.sort_values('nombre_dia')
        return daily
    
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_zone_passengers(df: pd.DataFrame) -> pd.DataFrame:
    if 'zona' not in df.columns or 'pasajeros' not in df.columns:
        return pd.DataFrame()
    
    zone = safe_groupby_aggregate(df, 'zona', 'pasajeros')
    if zone is not None:
        zone = zone.sort_values('total', ascending=False)
        return zone
    
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_line_passengers(df: pd.DataFrame) -> pd.DataFrame:
    if 'linea' not in df.columns or 'pasajeros' not in df.columns:
        return pd.DataFrame()
    
    line = safe_groupby_aggregate(df, 'linea', 'pasajeros')
    if line is not None:
        line = line.sort_values('total', ascending=False)
        return line
    
    return pd.DataFrame()


def get_peak_hours(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    hourly = get_hourly_passengers(df)
    if hourly is not None and len(hourly) > 0:
        peak = hourly.nlargest(top_n, 'total')
        return peak
    
    return pd.DataFrame()


def get_time_slot_passengers(df: pd.DataFrame) -> pd.DataFrame:
    if 'franja_horaria' not in df.columns or 'pasajeros' not in df.columns:
        return pd.DataFrame()
    
    time_slot = safe_groupby_aggregate(df, 'franja_horaria', 'pasajeros')
    if time_slot is not None:
        slot_order = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
        time_slot['franja_horaria'] = pd.Categorical(
            time_slot['franja_horaria'],
            categories=slot_order,
            ordered=True
        )
        time_slot = time_slot.sort_values('franja_horaria')
        return time_slot
    
    return pd.DataFrame()
