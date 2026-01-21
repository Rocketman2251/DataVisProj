import pandas as pd
import streamlit as st
from typing import Tuple, Optional, Dict, List
import numpy as np
import difflib
import config


# Mapping of possible column names (Spanish, English, variations)
COLUMN_MAPPINGS = {
    'fecha': ['fecha', 'date', 'fecha_servicio', 'service_date', 'day', 'dia', 'fecha_viaje', 'fecha_registro'],
    'hora': ['hora', 'hour', 'hora_servicio', 'service_hour', 'time', 'tiempo', 'hora_viaje', 'hora_registro'],
    'linea': ['linea', 'línea', 'line', 'route', 'ruta', 'linea_transporte', 'bus_line', 'linea_ruta', 'route_id'],
    'zona': ['zona', 'zone', 'area', 'region', 'sector', 'zona_geografica', 'location', 'barrio'],
    'pasajeros': ['pasajeros', 'passengers', 'num_pasajeros', 'pax', 'cantidad_pasajeros', 'count', 'personas', 'total_pasajeros'],
    'ocupacion': ['ocupacion', 'ocupación', 'occupancy', 'ocupacion_pct', 'ocupacion_porcentaje', 'capacity', 'porcentaje_ocupacion'],
    'retraso': ['retraso', 'retrasos', 'delay', 'delayminutos', 'delay_minutes', 'atraso', 'minutos_retraso', 'tiempo_retraso']
}


def find_similar_column(target: str, available: List[str], threshold: float = 0.5) -> Optional[str]:
    available_clean = {col: col.replace('_', '').replace(' ', '').lower() for col in available}
    target_clean = target.replace('_', '').replace(' ', '').lower()
    
    matches = difflib.get_close_matches(target_clean, available_clean.values(), n=1, cutoff=threshold)
    if matches:
        for col, clean in available_clean.items():
            if clean == matches[0]:
                return col
    return None


def standardize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.lower().str.strip()
    
    mapping = {}
    new_columns = {}
    available_cols = set(df_copy.columns)
    found_cols = set()
    
    # Exact and fuzzy matching
    for standard_name, variations in COLUMN_MAPPINGS.items():
        found = False
        
        # Try exact match
        for available_col in list(available_cols):
            col_clean = available_col.replace('_', '').replace(' ', '').replace('ó', 'o').lower()
            for variation in variations:
                var_clean = variation.replace('_', '').replace(' ', '').replace('ó', 'o').lower()
                if col_clean == var_clean:
                    if available_col != standard_name:
                        new_columns[available_col] = standard_name
                        mapping[available_col] = standard_name
                    found_cols.add(standard_name)
                    found = True
                    break
            if found:
                break
        
        # Try fuzzy matching
        if not found:
            similar = find_similar_column(standard_name, list(available_cols - found_cols), threshold=0.5)
            if similar:
                new_columns[similar] = standard_name
                mapping[similar] = standard_name
                found_cols.add(standard_name)
    
    if new_columns:
        df_copy = df_copy.rename(columns=new_columns)
    
    missing = [col for col in COLUMN_MAPPINGS.keys() if col not in found_cols and col not in df_copy.columns]
    
    return df_copy, mapping, missing


@st.cache_data(ttl=3600, show_spinner=False)
def load_csv_file(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        # Try different encodings and delimiters
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        delimiters = [',', ';', '\t', '|']
        df = None
        
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=delimiter)
                    if len(df) > 0 and len(df.columns) > 1:
                        break
                except Exception:
                    continue
            if df is not None and len(df) > 0:
                break
        
        if df is None or len(df) == 0:
            st.error("No se pudo cargar el archivo con las codificaciones disponibles", icon=config.ICON_ERROR)
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}", icon=config.ICON_ERROR)
        return None


def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str, Dict]:
    if df is None or len(df) == 0:
        return False, "El archivo está vacío", {}
    
    # Standardize columns
    df_mapped, mapping, missing = standardize_column_names(df)
    
    metadata = {
        'mapping': mapping,
        'missing': missing,
        'mapped_df': df_mapped,
        'generated': [],
        'original_columns': list(df.columns),
        'found_columns': list(df_mapped.columns)
    }
    
    # Generate missing columns if needed
    if missing:
        st.warning(f"Columnas no encontradas: {', '.join(missing)}", icon=config.ICON_WARNING)
        st.info("Se generarán datos sintéticos para completar el análisis", icon=config.ICON_INFO)
        st.info(f"Columnas encontradas: {', '.join([v for v in mapping.values()])}", icon=config.ICON_CHECK)
        
        df_mapped = generate_missing_columns(df_mapped, missing)
        metadata['mapped_df'] = df_mapped
        metadata['generated'] = missing
        
        return True, f"Datos procesados. Columnas generadas: {', '.join(missing)}", metadata
    
    if mapping:
        mapped_info = ", ".join([f"{k}->'{v}'" for k, v in list(mapping.items())[:3]])
        if len(mapping) > 3:
            mapped_info += f" ... (+{len(mapping)-3} más)"
        msg = f"Formato detectado correctamente. Columnas: {', '.join(list(df_mapped.columns)[:5])}..."
    else:
        msg = "Estructura validada sin cambios"
    
    return True, msg, metadata


def generate_missing_columns(df: pd.DataFrame, missing_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    np.random.seed(42)
    
    for col in missing_cols:
        if col == 'fecha':
            df['fecha'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        elif col == 'hora':
            df['hora'] = np.random.randint(0, 24, size=len(df))
        elif col == 'linea':
            lines = [f'L{i}' for i in range(1, min(11, len(df) // 30 + 1))]
            df['linea'] = np.random.choice(lines, size=len(df))
        elif col == 'zona':
            zones = ['Centro', 'Norte', 'Sur', 'Este', 'Oeste', 'Noroeste', 'Noreste']
            df['zona'] = np.random.choice(zones, size=len(df))
        elif col == 'pasajeros':
            df['pasajeros'] = np.random.randint(20, 200, size=len(df))
        elif col == 'ocupacion':
            df['ocupacion'] = np.random.uniform(15, 95, size=len(df))
        elif col == 'retraso':
            df['retraso'] = np.random.exponential(scale=3, size=len(df))
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
        'data_types': df.dtypes.to_dict(),
        'column_names': list(df.columns)
    }


def validate_data_quality(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    warnings = []
    
    if len(df) == 0:
        return False, ["El archivo está vacío"]
    
    # Check excessive missing values
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 50:
            warnings.append(f"'{col}' tiene {missing_pct:.1f}% valores faltantes")
    
    # Validate columns
    if 'hora' in df.columns:
        invalid_h = pd.to_numeric(df['hora'], errors='coerce').isnull().sum()
        if invalid_h > 0:
            warnings.append(f"'{invalid_h}' horas inválidas (se normalizarán)")
    
    if 'ocupacion' in df.columns:
        try:
            occ = pd.to_numeric(df['ocupacion'], errors='coerce')
            # Count values outside range (0-100), excluding NaN
            invalid_o = ((occ < 0) | (occ > 100)).fillna(False).sum()
            if int(invalid_o) > 0:
                warnings.append(f"'{int(invalid_o)}' valores de ocupación fuera del rango 0-100%")
        except Exception:
            pass
    
    return len(warnings) == 0, warnings


def generate_sample_csv_template() -> str:
    import io
    
    # Create example data
    template_data = {
        'fecha': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'hora': [6, 8, 14, 7, 15],
        'linea': ['L1', 'L2', 'L1', 'L3', 'L2'],
        'zona': ['Centro', 'Norte', 'Sur', 'Este', 'Oeste'],
        'pasajeros': [145, 203, 178, 92, 156],
        'ocupacion': [65.5, 78.2, 61.3, 42.1, 71.8],
        'retraso': [3, 5, 0, 2, 7]
    }
    
    df_template = pd.DataFrame(template_data)
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    df_template.to_csv(csv_buffer, index=False, encoding='utf-8')
    
    return csv_buffer.getvalue()


def get_sample_csv_bytes() -> bytes:
    csv_string = generate_sample_csv_template()
    return csv_string.encode('utf-8')
