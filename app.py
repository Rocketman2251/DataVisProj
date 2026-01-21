import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import plotly.express as px
import plotly.graph_objects as go

# Import modules
import config
from modules import data_loader, preprocessing
from modules import flow_analysis, occupancy_analysis, delay_analysis
from modules import visualization

# Configure Streamlit page
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None


def main():
    
    # Header
    st.markdown(f'<h1 class="main-title"><i class="fas fa-bus"></i> Sistema de Análisis y Visualización de Transporte Urbano</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Análisis integral del transporte público urbano</p>', unsafe_allow_html=True)
    st.divider()
    
    # Sidebar for file upload and filters
    with st.sidebar:
        st.header(f"{config.ICON_SETTINGS} Configuración")
        
        # File upload section
        st.subheader(f"{config.ICON_UPLOAD} Cargar Datos")
        
        # Help section with format info
        with st.expander("¿Qué formato de CSV necesito?", icon=config.ICON_INFO):
            st.markdown("""
            **Columnas requeridas:**
            - `fecha` - Fecha del servicio (YYYY-MM-DD)
            - `hora` - Hora del día (0-23)
            - `linea` - Identificador de línea (ej: L1, L2)
            - `zona` - Zona geográfica (ej: Centro, Norte)
            - `pasajeros` - Número de pasajeros (número entero)
            - `ocupacion` - Porcentaje de ocupación (0-100)
            - `retraso` - Minutos de retraso (número entero)
            
            **El sistema aceptará:**
            - Columnas con nombres en español o inglés
            - Variaciones de nombres (ej: "Pasajeros", "num_pasajeros", "pax")
            - Generará columnas faltantes automáticamente
            """)
            
            # Download template button
            template_csv = data_loader.get_sample_csv_bytes()
            st.download_button(
                label="Descargar Template CSV",
                data=template_csv,
                file_name="template_transporte.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        uploaded_file = st.file_uploader(
            "Cargar archivo CSV",
            type=['csv'],
            help="Seleccione un archivo CSV con datos de transporte urbano"
        )
        
        if uploaded_file is not None:
            try:
                # Load CSV with intelligent column detection
                df = data_loader.load_csv_file(uploaded_file)
                
                if df is not None:
                    # Validate and map columns
                    is_valid, message, metadata = data_loader.validate_csv_structure(df)
                    
                    if is_valid:
                        st.success(message)
                        
                        # Use mapped dataframe
                        df_mapped = metadata['mapped_df']
                        generated = metadata['generated']
                        mapping = metadata.get('mapping', {})
                        
                        # Show detailed information
                        with st.expander("Detalle de Carga", expanded=False, icon=config.ICON_STATS):
                            st.markdown("**Columnas originales en archivo:**")
                            st.code(', '.join(metadata.get('original_columns', [])))
                            
                            if mapping:
                                st.markdown("**Mapeo de columnas:**")
                                for required, found in mapping.items():
                                    st.caption(f"  '{required}' ← '{found}'")
                            
                            if generated:
                                st.markdown(f"**Columnas generadas ({len(generated)}):**")
                                st.warning(f"{', '.join(generated)}")
                        
                        # Validate data quality
                        quality_ok, quality_warnings = data_loader.validate_data_quality(df_mapped)
                        if quality_warnings:
                            with st.expander("Advertencias de Calidad", expanded=False, icon=config.ICON_WARNING):
                                for w in quality_warnings:
                                    st.warning(w, icon=config.ICON_WARNING)

                        

                        # Process data
                        df_processed = preprocessing.preprocess_full_pipeline(df_mapped)
                        
                        st.session_state.df = df_mapped
                        st.session_state.df_processed = df_processed
                        
                        info = data_loader.get_data_info(df_mapped)
                        st.success(f"Datos cargados: {info['rows']} registros | {info['columns']} columnas", icon=config.ICON_CHECK)
                    else:
                        st.error(f"{message}", icon=config.ICON_ERROR)
            
            except Exception as e:
                st.error(f"Error procesando archivo: {str(e)}", icon=config.ICON_ERROR)
        
        st.divider()
        
        # Filters section
        if st.session_state.df_processed is not None:
            st.subheader(f"{config.ICON_FILTER} Filtros de Análisis")
            
            df_processed = st.session_state.df_processed
            
            # Date range filter
            with st.expander("Rango de Fechas", expanded=True, icon=config.ICON_TIME):
                min_date = pd.to_datetime(df_processed['fecha']).min().date()
                max_date = pd.to_datetime(df_processed['fecha']).max().date()
                
                date_range = st.slider(
                    "Seleccione rango de fechas",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date),
                    format="YYYY-MM-DD"
                )
            
            # Line filter
            with st.expander("Línea de Transporte", icon=config.ICON_BUS):
                lines = sorted(df_processed['linea'].unique())
                selected_lines = st.multiselect(
                    "Seleccione líneas",
                    options=lines,
                    default=lines[:3] if len(lines) > 0 else lines
                )
            
            # Zone filter
            with st.expander("Zona Geográfica", icon=config.ICON_MAP):
                zones = sorted(df_processed['zona'].unique())
                selected_zones = st.multiselect(
                    "Seleccione zonas",
                    options=zones,
                    default=zones
                )
            
            # Day of week filter
            with st.expander("Día de la Semana", icon=config.ICON_TIME):
                day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                selected_days = st.multiselect(
                    "Seleccione días",
                    options=day_names,
                    default=day_names
                )
            
            # Time slot filter
            with st.expander("Franja Horaria", icon=config.ICON_TIME):
                time_slots = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
                selected_slots = st.multiselect(
                    "Seleccione franjas horarias",
                    options=time_slots,
                    default=time_slots
                )
            
            # Apply filters
            # Convert date_range values to datetime for comparison
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            
            filtered_df = df_processed[
                (pd.to_datetime(df_processed['fecha']).dt.date >= date_range[0]) &
                (pd.to_datetime(df_processed['fecha']).dt.date <= date_range[1]) &
                (df_processed['linea'].isin(selected_lines)) &
                (df_processed['zona'].isin(selected_zones)) &
                (df_processed['nombre_dia'].isin([
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                ])) &
                (df_processed['franja_horaria'].isin(selected_slots))
            ]
            
            # Map day names to Spanish
            day_mapping = {
                'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
            }
            filtered_df['nombre_dia_es'] = filtered_df['nombre_dia'].map(day_mapping)
            
            filtered_df = filtered_df[
                filtered_df['nombre_dia_es'].isin(selected_days)
            ]
            
            # Summary statistics in sidebar
            st.divider()
            st.subheader(f"{config.ICON_STATS} Resumen de Datos")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Registros", f"{len(filtered_df):,}")
                st.metric("Fechas", f"{filtered_df['fecha'].nunique()}")
            with col2:
                st.metric("Líneas", f"{filtered_df['linea'].nunique()}")
                st.metric("Zonas", f"{filtered_df['zona'].nunique()}")
        
        else:
            filtered_df = None
            st.info("Cargue un archivo CSV para comenzar", icon=config.ICON_INFO)
    
    # Main content
    if st.session_state.df_processed is not None and filtered_df is not None and len(filtered_df) > 0:
        
        # Dashboard sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            f"{config.ICON_DASHBOARD} Dashboard Principal",
            f"{config.ICON_GROUP} Flujo de Pasajeros",
            f"{config.ICON_BUS} Ocupación del Sistema",
            f"{config.ICON_TIME} Análisis de Retrasos",
            f"{config.ICON_STATS} Análisis Integrado"
        ])
        
        # ==================== TAB 1: Main Dashboard ====================
        with tab1:
            st.header("Dashboard Principal - KPIs")
            
            # Calculate KPIs
            total_passengers = filtered_df['pasajeros'].sum()
            avg_occupancy = filtered_df['ocupacion'].mean()
            avg_delay = filtered_df['retraso'].mean()
            punctuality = delay_analysis.get_punctuality_rate(filtered_df)
            punctuality_rate = punctuality['porcentaje']
            
            # Display KPI cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total de Pasajeros",
                    value=f"{total_passengers:,.0f}"
                )
            
            with col2:
                st.metric(
                    label="Ocupación Promedio",
                    value=f"{avg_occupancy:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="Retraso Promedio",
                    value=f"{avg_delay:.1f} min"
                )
            
            with col4:
                st.metric(
                    label="Tasa de Puntualidad",
                    value=f"{punctuality_rate:.1f}%"
                )
            
            st.divider()
            
            # Additional metrics
            st.subheader("Métricas Detalladas")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Servicios Totales", f"{len(filtered_df):,}")
            
            with col2:
                st.metric("Líneas Activas", f"{filtered_df['linea'].nunique()}")
            
            with col3:
                st.metric("Zonas Cubiertas", f"{filtered_df['zona'].nunique()}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Retraso Máximo", f"{filtered_df['retraso'].max():.1f} min")
            
            with col2:
                st.metric("Ocupación Máxima", f"{filtered_df['ocupacion'].max():.1f}%")
            
            with col3:
                st.metric("Pasajeros Mínimos por Servicio", f"{filtered_df['pasajeros'].min():.0f}")
            
            st.divider()
            
            # Summary statistics by category
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Pasajeros por Tipo de Día")
                passengers_by_type = filtered_df.groupby('tipo_dia')['pasajeros'].agg(['sum', 'mean'])
                passengers_by_type.columns = ['Total', 'Promedio']
                st.dataframe(passengers_by_type, width='stretch')
            
            with col2:
                st.subheader("Retrasos por Tipo de Día")
                delays_by_type = filtered_df.groupby('tipo_dia')['retraso'].agg(['mean', 'std', 'max'])
                delays_by_type.columns = ['Promedio', 'Desv. Est.', 'Máximo']
                st.dataframe(delays_by_type, width='stretch')
        
        # ==================== TAB 2: Passenger Flow ====================
        with tab2:
            st.header("Análisis de Flujo de Pasajeros")
            
            # Hourly passengers
            st.subheader("Pasajeros por Hora del Día")
            hourly_data = flow_analysis.get_hourly_passengers(filtered_df)
            fig_hourly = visualization.create_line_chart(
                hourly_data, 'hora', 'total',
                'Flujo de Pasajeros por Hora',
                'Hora del Día', 'Total de Pasajeros'
            )
            st.plotly_chart(fig_hourly, width='stretch')
            
            col1, col2 = st.columns(2)
            
            # Daily passengers
            with col1:
                st.subheader("Pasajeros por Día de la Semana")
                daily_data = flow_analysis.get_daily_passengers(filtered_df)
                day_mapping = {
                    'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                }
                daily_data['nombre_dia'] = daily_data['nombre_dia'].map(day_mapping)
                
                fig_daily = visualization.create_bar_chart(
                    daily_data, 'nombre_dia', 'total',
                    'Flujo de Pasajeros por Día',
                    'Día de la Semana', 'Total de Pasajeros'
                )
                st.plotly_chart(fig_daily, width='stretch')
            
            # Zone passengers
            with col2:
                st.subheader("Pasajeros por Zona Geográfica")
                zone_data = flow_analysis.get_zone_passengers(filtered_df)
                fig_zone = visualization.create_bar_chart(
                    zone_data, 'zona', 'total',
                    'Flujo de Pasajeros por Zona',
                    'Zona', 'Total de Pasajeros'
                )
                st.plotly_chart(fig_zone, width='stretch')
            
            # Peak hours
            st.subheader("Horarios Pico")
            peak_hours = flow_analysis.get_peak_hours(filtered_df, top_n=5)
            st.dataframe(
                peak_hours[['hora', 'total', 'promedio']].rename(
                    columns={'hora': 'Hora', 'total': 'Total Pasajeros', 'promedio': 'Promedio'}
                ),
                width='stretch',
                hide_index=True
            )
            
            # Time slot analysis
            st.subheader("Pasajeros por Franja Horaria")
            slot_data = flow_analysis.get_time_slot_passengers(filtered_df)
            fig_slot = visualization.create_bar_chart(
                slot_data, 'franja_horaria', 'total',
                'Flujo de Pasajeros por Franja Horaria',
                'Franja Horaria', 'Total de Pasajeros'
            )
            st.plotly_chart(fig_slot, width='stretch')
            
            # Lines analysis
            st.subheader("Pasajeros por Línea de Transporte")
            line_data = flow_analysis.get_line_passengers(filtered_df)
            fig_line = visualization.create_bar_chart(
                line_data.head(10), 'linea', 'total',
                'Top 10 Líneas por Pasajeros',
                'Línea', 'Total de Pasajeros'
            )
            st.plotly_chart(fig_line, width='stretch')
        
        # ==================== TAB 3: Occupancy Analysis ====================
        with tab3:
            st.header("Análisis de Ocupación del Sistema")
            
            col1, col2 = st.columns(2)
            
            # Occupancy by hour
            with col1:
                st.subheader("Ocupación Promedio por Hora")
                hourly_occ = occupancy_analysis.get_occupancy_by_hour(filtered_df)
                fig_hour_occ = visualization.create_line_chart(
                    hourly_occ, 'hora', 'promedio',
                    'Ocupación por Hora del Día',
                    'Hora', 'Ocupación (%)'
                )
                st.plotly_chart(fig_hour_occ, width='stretch')
            
            # Occupancy by day
            with col2:
                st.subheader("Ocupación Promedio por Día")
                daily_occ = occupancy_analysis.get_occupancy_by_day(filtered_df)
                day_mapping = {
                    'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                }
                daily_occ['nombre_dia'] = daily_occ['nombre_dia'].map(day_mapping)
                
                fig_day_occ = visualization.create_bar_chart(
                    daily_occ, 'nombre_dia', 'promedio',
                    'Ocupación por Día',
                    'Día de la Semana', 'Ocupación (%)'
                )
                st.plotly_chart(fig_day_occ, width='stretch')
            
            # Occupancy heatmap
            st.subheader("Mapa de Calor: Ocupación por Día y Hora")
            heatmap_data = occupancy_analysis.get_occupancy_heatmap(filtered_df)
            fig_heatmap = visualization.create_heatmap(
                heatmap_data,
                'Ocupación por Día y Hora',
                'Día de la Semana',
                'Hora del Día'
            )
            st.plotly_chart(fig_heatmap, width='stretch')
            
            col1, col2 = st.columns(2)
            
            # Occupancy by line
            with col1:
                st.subheader("Ocupación Promedio por Línea")
                line_occ = occupancy_analysis.get_occupancy_by_line(filtered_df)
                fig_line_occ = visualization.create_bar_chart(
                    line_occ.head(10), 'linea', 'promedio',
                    'Top 10 Líneas Más Congestionadas',
                    'Línea', 'Ocupación Promedio (%)'
                )
                st.plotly_chart(fig_line_occ, width='stretch')
            
            # Occupancy by zone
            with col2:
                st.subheader("Ocupación Promedio por Zona")
                zone_occ = occupancy_analysis.get_occupancy_by_zone(filtered_df)
                fig_zone_occ = visualization.create_bar_chart(
                    zone_occ, 'zona', 'promedio',
                    'Ocupación por Zona Geográfica',
                    'Zona', 'Ocupación Promedio (%)'
                )
                st.plotly_chart(fig_zone_occ, width='stretch')
            
            # Time slot occupancy
            st.subheader("Ocupación por Franja Horaria")
            slot_occ = occupancy_analysis.get_occupancy_by_time_slot(filtered_df)
            fig_slot_occ = visualization.create_bar_chart(
                slot_occ, 'franja_horaria', 'promedio',
                'Ocupación por Franja Horaria',
                'Franja Horaria', 'Ocupación (%)'
            )
            st.plotly_chart(fig_slot_occ, width='stretch')
        
        # ==================== TAB 4: Delay Analysis ====================
        with tab4:
            st.header("Análisis de Retrasos")
            
            # Delay statistics
            delay_stats = delay_analysis.get_delay_statistics(filtered_df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Retraso Promedio", f"{delay_stats['promedio']:.2f} min")
            
            with col2:
                st.metric("Retraso Mediana", f"{delay_stats['mediana']:.2f} min")
            
            with col3:
                st.metric("Desv. Estándar", f"{delay_stats['desv_est']:.2f} min")
            
            with col4:
                st.metric("Retraso Máximo", f"{delay_stats['maximo']:.2f} min")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            # Delay histogram
            with col1:
                st.subheader("Distribución de Retrasos")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=filtered_df['retraso'],
                    nbinsx=30,
                    name='Retrasos'
                ))
                fig_hist.update_layout(
                    title='Histograma de Retrasos',
                    xaxis_title='Retraso (minutos)',
                    yaxis_title='Frecuencia',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_hist, width='stretch')
            
            # Occupancy histogram
            with col2:
                st.subheader("Distribución de Ocupación")
                fig_hist_occ = go.Figure()
                fig_hist_occ.add_trace(go.Histogram(
                    x=filtered_df['ocupacion'],
                    nbinsx=30,
                    name='Ocupación',
                    marker=dict(color='#FF6B35')
                ))
                fig_hist_occ.update_layout(
                    title='Histograma de Ocupación',
                    xaxis_title='Ocupación (%)',
                    yaxis_title='Frecuencia',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_hist_occ, width='stretch')
            
            # Delay by hour
            st.subheader("Retraso Promedio por Hora")
            hourly_delay = delay_analysis.get_delay_by_hour(filtered_df)
            fig_hour_delay = visualization.create_line_chart(
                hourly_delay, 'hora', 'promedio',
                'Retraso por Hora del Día',
                'Hora', 'Retraso Promedio (min)'
            )
            st.plotly_chart(fig_hour_delay, width='stretch')
            
            col1, col2 = st.columns(2)
            
            # Delay by line
            with col1:
                st.subheader("Retraso Promedio por Línea")
                line_delay = delay_analysis.get_delay_by_line(filtered_df)
                fig_line_delay = visualization.create_bar_chart(
                    line_delay.head(10), 'linea', 'promedio',
                    'Top 10 Líneas con Mayor Retraso',
                    'Línea', 'Retraso Promedio (min)'
                )
                st.plotly_chart(fig_line_delay, width='stretch')
            
            # Delay by zone
            with col2:
                st.subheader("Retraso Promedio por Zona")
                zone_delay = delay_analysis.get_delay_by_zone(filtered_df)
                fig_zone_delay = visualization.create_bar_chart(
                    zone_delay, 'zona', 'promedio',
                    'Retraso por Zona Geográfica',
                    'Zona', 'Retraso Promedio (min)'
                )
                st.plotly_chart(fig_zone_delay, width='stretch')
            
            # Delay by day
            st.subheader("Retraso Promedio por Día de la Semana")
            daily_delay = delay_analysis.get_delay_by_day(filtered_df)
            day_mapping = {
                'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
            }
            daily_delay['nombre_dia'] = daily_delay['nombre_dia'].map(day_mapping)
            
            fig_day_delay = visualization.create_bar_chart(
                daily_delay, 'nombre_dia', 'promedio',
                'Retraso por Día de la Semana',
                'Día', 'Retraso Promedio (min)'
            )
            st.plotly_chart(fig_day_delay, width='stretch')
            
            # Punctuality metrics
            st.subheader("Indicadores de Puntualidad")
            punct_stats = delay_analysis.get_punctuality_rate(filtered_df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Servicios Puntuales", f"{punct_stats['puntual']:,}")
            
            with col2:
                st.metric("Servicios Retrasados", f"{punct_stats['retrasada']:,}")
            
            with col3:
                st.metric("Tasa de Puntualidad", f"{punct_stats['porcentaje']:.1f}%")
            
            # Punctuality by line
            st.subheader("Puntualidad por Línea")
            punct_by_line = delay_analysis.get_punctuality_by_line(filtered_df)
            fig_punct = visualization.create_bar_chart(
                punct_by_line.head(10), 'linea', 'porcentaje',
                'Top 10 Líneas Más Puntuales',
                'Línea', 'Porcentaje Puntual (%)'
            )
            st.plotly_chart(fig_punct, width='stretch')
        
        # ==================== TAB 5: Integrated Analysis ====================
        with tab5:
            st.header("Análisis Integrado")
            
            # Scatter: Occupancy vs Delay
            st.subheader("Relación: Ocupación vs Retrasos")
            fig_scatter = visualization.create_scatter_plot(
                filtered_df, 'ocupacion', 'retraso',
                'Ocupación vs Retrasos por Servicio',
                'Ocupación (%)', 'Retraso (minutos)',
                color_col='linea'
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, width='stretch')
            
            # Critical lines identification
            st.subheader("Líneas Críticas (Alta Ocupación + Alto Retraso)")
            critical_lines = occupancy_analysis.get_critical_lines(filtered_df, threshold_occupancy=75)
            
            if len(critical_lines) > 0:
                st.warning(f"⚠️ Se identificaron {len(critical_lines)} líneas críticas")
                st.dataframe(
                    critical_lines.rename(columns={
                        'linea': 'Línea',
                        'ocupacion_promedio': 'Ocupación Promedio (%)',
                        'retraso_promedio': 'Retraso Promedio (min)',
                        'pasajeros_total': 'Total Pasajeros'
                    }),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.success("✓ No se identificaron líneas críticas")
            
            # Line performance matrix
            st.subheader("Matriz de Rendimiento por Línea")
            
            line_performance = filtered_df.groupby('linea').agg({
                'pasajeros': 'sum',
                'ocupacion': 'mean',
                'retraso': 'mean',
                'es_puntual': lambda x: (x.sum() / len(x) * 100)
            })
            
            line_performance.columns = ['Pasajeros', 'Ocupación (%)', 'Retraso (min)', 'Puntualidad (%)']
            line_performance = line_performance.sort_values('Pasajeros', ascending=False).head(15)
            
            st.dataframe(line_performance, width='stretch')
            
            # Zone performance
            st.subheader("Matriz de Rendimiento por Zona")
            
            zone_performance = filtered_df.groupby('zona').agg({
                'pasajeros': 'sum',
                'ocupacion': 'mean',
                'retraso': 'mean',
                'es_puntual': lambda x: (x.sum() / len(x) * 100)
            })
            
            zone_performance.columns = ['Pasajeros', 'Ocupación (%)', 'Retraso (min)', 'Puntualidad (%)']
            zone_performance = zone_performance.sort_values('Pasajeros', ascending=False)
            
            st.dataframe(zone_performance, width='stretch')
            
            # Correlation heatmap
            st.subheader("Correlación entre Variables")
            
            correlation_data = filtered_df[['pasajeros', 'ocupacion', 'retraso']].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_data.values,
                x=['Pasajeros', 'Ocupación', 'Retraso'],
                y=['Pasajeros', 'Ocupación', 'Retraso'],
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                text=correlation_data.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))
            fig_corr.update_layout(
                title='Matriz de Correlación',
                height=400
            )
            st.plotly_chart(fig_corr, width='stretch')
        
        # Export section
        st.divider()
        st.header(f"{config.ICON_DOWNLOAD} Descarga de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download filtered data as CSV
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Descargar Datos Filtrados (CSV)",
                data=csv_data,
                file_name=f"transporte_urbano_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                icon=config.ICON_DOWNLOAD
            )
        
        with col2:
            # Download summary statistics
            summary_data = {
                'Métrica': [
                    'Total de Pasajeros',
                    'Ocupación Promedio (%)',
                    'Retraso Promedio (min)',
                    'Tasa de Puntualidad (%)',
                    'Número de Servicios',
                    'Líneas Activas',
                    'Zonas Cubiertas'
                ],
                'Valor': [
                    f"{total_passengers:,.0f}",
                    f"{avg_occupancy:.2f}",
                    f"{avg_delay:.2f}",
                    f"{punctuality_rate:.2f}",
                    f"{len(filtered_df):,}",
                    f"{filtered_df['linea'].nunique()}",
                    f"{filtered_df['zona'].nunique()}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            csv_summary = summary_df.to_csv(index=False)
            
            st.download_button(
                label="Descargar Resumen (CSV)",
                data=csv_summary,
                file_name=f"resumen_transporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                icon=config.ICON_DOWNLOAD
            )
    
    else:
        if st.session_state.df_processed is not None and (filtered_df is None or len(filtered_df) == 0):
            st.warning("No hay datos que coincidan con los filtros seleccionados. Ajuste los filtros.", icon=config.ICON_WARNING)
        else:
            filtered_df = None
            st.markdown(config.LANDING_PAGE_HTML, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
