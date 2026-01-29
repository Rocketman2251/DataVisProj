"""
ML Integration Example for Streamlit App
Shows how to integrate ML anomaly detection into the existing app
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import custom ML modules (these would be in modules/ folder)
# from modules.ml_feature_engineering import TransportFeatureEngineer
# from modules.ml_anomaly_detection import TransportAnomalyDetector, AnomalyAnalyzer
# from modules.ml_classifier import AnomalyClassifier


def create_ml_tab(df: pd.DataFrame):
    """
    Create the ML Analysis tab in Streamlit app for DataVisProj
    
    Args:
        df: Preprocessed transport data DataFrame (already filtered)
    """
    st.header(":material/smart_toy: Análisis con Machine Learning")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.markdown("### :material/settings: Configuración ML")
    contamination = st.sidebar.slider(
        "Tasa de contaminación esperada (%)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Porcentaje estimado de anomalías en los datos"
    ) / 100
    
    run_analysis = st.sidebar.button(":material/smart_toy: Ejecutar Análisis ML", type="primary")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **¿Qué hace este análisis?**
        
        Este sistema de Machine Learning identifica automáticamente eventos anómalos en los datos de transporte:
        
        - :material/restaurant: **Horarios de almuerzo** de conductores
        - :material/celebration: **Eventos especiales** (conciertos, partidos, manifestaciones)
        - :material/cloud: **Condiciones climáticas** adversas
        - :material/car_crash: **Accidentes** de tráfico
        - :material/warning: **Paros o huelgas**
        
        Los datos limpios permiten análisis más precisos y decisiones mejor informadas.
        """)
    
    with col2:
        st.metric("Total de Registros", f"{len(df):,}")
        st.metric("Período", f"{df['fecha'].min()} - {df['fecha'].max()}")
    
    if run_analysis:
        with st.spinner(":material/sync: Ejecutando análisis ML... Esto puede tomar unos segundos."):
            # Step 1: Feature Engineering
            st.markdown("### 1️. Ingeniería de Características")
            progress_bar = st.progress(0)
            
            # Here you would call: engineer = TransportFeatureEngineer()
            # df_features, feature_names = engineer.engineer_features(df)
            # For demo purposes, we'll simulate this
            
            progress_bar.progress(25)
            st.success(":material/check_circle: Características creadas: 24 features temporales, estadísticas y contextuales")
            
            # Step 2: Anomaly Detection
            st.markdown("### 2️. Detección de Anomalías")
            progress_bar.progress(50)
            
            # Here you would call: detector = TransportAnomalyDetector(contamination=contamination)
            # detector.fit(X)
            # df_anomalies = detector.get_anomaly_info(df, X)
            
            # Simulate results
            n_anomalies = int(len(df) * contamination)
            st.success(f":material/check_circle: Detectadas {n_anomalies:,} anomalías ({contamination*100:.1f}% del total)")
            
            progress_bar.progress(75)
            
            # Step 3: Classification
            st.markdown("### 3️. Clasificación de Anomalías")
            
            # Simulate classification results
            anomaly_distribution = {
                'LUNCH_BREAK': 0.35,
                'SPECIAL_EVENT': 0.15,
                'WEATHER': 0.20,
                'ACCIDENT': 0.15,
                'STRIKE': 0.05,
                'NORMAL': 0.10
            }
            
            progress_bar.progress(100)
            st.success(":material/check_circle: Anomalías clasificadas en 6 categorías")
            
            # Display results
            st.markdown("---")
            st.markdown("## :material/bar_chart: Resultados del Análisis")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                ":material/summarize: Resumen", 
                ":material/schedule: Análisis Temporal",
                ":material/table_chart: Detalle de Anomalías",
                ":material/compare: Comparación Antes/Después"
            ])
            
            with tab1:
                show_summary_view(n_anomalies, anomaly_distribution)
            
            with tab2:
                show_temporal_analysis(df, n_anomalies)
            
            with tab3:
                show_anomaly_details(df, n_anomalies)
            
            with tab4:
                show_comparison_view(df)


def show_summary_view(n_anomalies: int, distribution: dict):
    """Show summary statistics and distribution"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Anomalías Detectadas", f"{n_anomalies:,}")
    with col2:
        st.metric("Tipo Más Común", "Horarios Almuerzo")
    with col3:
        st.metric("Confianza Promedio", "87.3%")
    with col4:
        st.metric("Precisión Modelo", "91.2%")
    
    st.markdown("### Distribución por Tipo de Anomalía")
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(distribution.keys()),
        values=list(distribution.values()),
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title="Distribución de Tipos de Anomalías",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    summary_df = pd.DataFrame({
        'Tipo de Anomalía': list(distribution.keys()),
        'Porcentaje': [f"{v*100:.1f}%" for v in distribution.values()],
        'Descripción': [
            'Reducción de servicio en horas de almuerzo',
            'Eventos masivos que alteran el servicio',
            'Condiciones climáticas adversas',
            'Accidentes que bloquean rutas',
            'Paros o huelgas laborales',
            'Operación normal sin anomalías'
        ]
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def show_temporal_analysis(df: pd.DataFrame, n_anomalies: int):
    """Show temporal patterns in anomalies"""
    
    st.markdown("### Patrones Temporales de Anomalías")
    
    # Simulate hourly distribution
    hourly_anomalies = {
        str(h): np.random.randint(0, n_anomalies//20) 
        for h in range(24)
    }
    # Peak during lunch hours
    for h in [12, 13, 14]:
        hourly_anomalies[str(h)] = int(n_anomalies * 0.15)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(hourly_anomalies.keys()),
        y=list(hourly_anomalies.values()),
        name='Anomalías',
        marker_color='indianred'
    ))
    
    fig.update_layout(
        title='Distribución de Anomalías por Hora del Día',
        xaxis_title='Hora',
        yaxis_title='Número de Anomalías',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Por Día de la Semana")
        daily_data = pd.DataFrame({
            'Día': ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
            'Anomalías': [120, 115, 118, 122, 125, 80, 70]
        })
        fig = px.bar(daily_data, x='Día', y='Anomalías', color='Anomalías')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Por Línea de Transporte")
        line_data = pd.DataFrame({
            'Línea': ['L1', 'L2', 'L3', 'L4', 'L5'],
            'Anomalías': [85, 92, 78, 95, 88]
        })
        fig = px.bar(line_data, x='Línea', y='Anomalías', color='Anomalías')
        st.plotly_chart(fig, use_container_width=True)


def show_anomaly_details(df: pd.DataFrame, n_anomalies: int):
    """Show detailed table of anomalies"""
    
    st.markdown("### Tabla Detallada de Anomalías")
    
    # Create sample anomaly data
    sample_anomalies = pd.DataFrame({
        'Fecha': pd.date_range('2024-01-01', periods=10),
        'Hora': [12, 13, 14, 8, 17, 13, 12, 19, 14, 13],
        'Línea': ['L1', 'L2', 'L1', 'L3', 'L2', 'L4', 'L1', 'L5', 'L3', 'L2'],
        'Zona': ['Centro', 'Norte', 'Centro', 'Sur', 'Este', 'Centro', 'Centro', 'Oeste', 'Sur', 'Norte'],
        'Tipo Anomalía': ['LUNCH_BREAK', 'LUNCH_BREAK', 'LUNCH_BREAK', 'ACCIDENT', 
                          'SPECIAL_EVENT', 'LUNCH_BREAK', 'LUNCH_BREAK', 'WEATHER',
                          'LUNCH_BREAK', 'LUNCH_BREAK'],
        'Confianza': [92.3, 88.5, 91.2, 76.8, 82.4, 89.1, 93.5, 71.2, 87.9, 90.3],
        'Pasajeros': [45, 52, 48, 35, 185, 50, 43, 68, 51, 49],
        'Ocupación': [32.1, 38.5, 35.2, 25.8, 92.3, 36.7, 31.5, 48.2, 37.1, 35.9]
    })
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tipo_filter = st.multiselect(
            "Filtrar por tipo:",
            options=['LUNCH_BREAK', 'SPECIAL_EVENT', 'WEATHER', 'ACCIDENT', 'STRIKE'],
            default=['LUNCH_BREAK', 'SPECIAL_EVENT', 'WEATHER', 'ACCIDENT']
        )
    
    with col2:
        confianza_min = st.slider("Confianza mínima:", 0, 100, 70)
    
    with col3:
        top_n = st.number_input("Mostrar top N:", min_value=10, max_value=100, value=20)
    
    # Apply filters
    filtered = sample_anomalies[
        (sample_anomalies['Tipo Anomalía'].isin(tipo_filter)) &
        (sample_anomalies['Confianza'] >= confianza_min)
    ].head(top_n)
    
    # Display table
    st.dataframe(
        filtered.style.background_gradient(subset=['Confianza'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=":material/download: Descargar anomalías (CSV)",
        data=csv,
        file_name=f"anomalias_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def show_comparison_view(df: pd.DataFrame):
    """Show before/after comparison"""
    
    st.markdown("### Comparación: Datos Originales vs Limpios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### :material/analytics: Con Anomalías")
        original_stats = {
            'Métrica': ['Pasajeros Promedio', 'Ocupación Promedio', 'Retraso Promedio', 'Desv. Estándar Ocupación'],
            'Valor': ['142 pasajeros', '68.5%', '7.2 min', '18.3%']
        }
        st.dataframe(pd.DataFrame(original_stats), hide_index=True)
    
    with col2:
        st.markdown("#### :material/auto_fix_high: Sin Anomalías")
        clean_stats = {
            'Métrica': ['Pasajeros Promedio', 'Ocupación Promedio', 'Retraso Promedio', 'Desv. Estándar Ocupación'],
            'Valor': ['156 pasajeros', '72.1%', '5.8 min', '12.1%']
        }
        st.dataframe(pd.DataFrame(clean_stats), hide_index=True)
    
    st.success("""
    **:material/lightbulb: Insights:** Al remover las anomalías, observamos que:
    - El sistema realmente opera con mayor capacidad de lo que parece
    - La variabilidad real es menor, facilitando la predicción
    - Los retrasos estructurales son menores que los aparentes
    """)
    
    # Visualization comparison
    st.markdown("#### Visualización Comparativa")
    
    hours = list(range(24))
    with_anomalies = [np.random.randint(80, 150) for _ in hours]
    # Reduce values during lunch hours for "with anomalies"
    for h in [12, 13, 14]:
        with_anomalies[h] = np.random.randint(40, 60)
    
    without_anomalies = [np.random.randint(90, 160) for _ in hours]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=with_anomalies,
        name='Con Anomalías',
        mode='lines+markers',
        line=dict(color='lightcoral', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=without_anomalies,
        name='Sin Anomalías',
        mode='lines+markers',
        line=dict(color='lightgreen', width=2)
    ))
    
    fig.update_layout(
        title='Pasajeros por Hora: Comparación',
        xaxis_title='Hora del Día',
        yaxis_title='Promedio de Pasajeros',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    st.set_page_config(page_title="ML Analysis Demo", layout="wide")
    
    # Demo with sample data
    st.title(":material/smart_toy: Demo: Integración ML para Detección de Anomalías")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000)
    sample_data = pd.DataFrame({
        'fecha': dates,
        'hora': np.random.randint(0, 24, 1000),
        'linea': np.random.choice(['L1', 'L2', 'L3'], 1000),
        'zona': np.random.choice(['Centro', 'Norte', 'Sur'], 1000),
        'pasajeros': np.random.randint(50, 200, 1000),
        'ocupacion': np.random.uniform(30, 90, 1000),
        'retraso': np.random.uniform(0, 15, 1000)
    })
    
    create_ml_tab(sample_data)
