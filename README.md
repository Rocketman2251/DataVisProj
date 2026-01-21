# Sistema de Análisis y Visualización de Transporte Urbano

## Urban Transport Analysis and Visualization System

Una aplicación web completa para análisis integral del transporte público urbano con visualizaciones interactivas, dashboards en tiempo real y análisis estadísticos.

### Características Principales
**Dashboard Principal con KPIs**
- Total de pasajeros
- Ocupación promedio del sistema
- Retraso promedio
- Tasa de puntualidad

**Análisis de Flujo de Pasajeros**
- Distribución horaria de pasajeros
- Análisis por día de la semana
- Análisis geográfico por zonas
- Identificación de horarios pico
- Análisis por franja horaria

**Análisis de Ocupación**
- Ocupación promedio por línea
- Ocupación promedio por zona
- Matriz de ocupación (día x hora)
- Líneas críticas (más congestionadas)
- Evolución temporal de ocupación

**Análisis de Retrasos**
- Distribución de retrasos (histograma)
- Retraso promedio por línea
- Retraso promedio por zona
- Retraso por hora del día
- Indicadores de puntualidad

**Análisis Integrado**
- Relación ocupación vs retrasos
- Identificación de líneas críticas
- Matriz de rendimiento por línea
- Matriz de rendimiento por zona
- Análisis de correlación

**Funcionalidades de Exportación**
- Descarga de datos filtrados en CSV
- Descarga de resumen de análisis
- Exportación de gráficos

### Estructura del Proyecto

```
DataProj/
├── app.py                           # Aplicación principal Streamlit
├── requirements.txt                 # Dependencias Python
├── sample_data.csv                  # Datos de ejemplo para pruebas
├── README.md                        # Este archivo
└── modules/
    ├── __init__.py
    ├── data_loader.py               # Carga y validación de datos
    ├── preprocessing.py             # Limpieza y transformación de datos
    ├── flow_analysis.py             # Análisis de flujo de pasajeros
    ├── occupancy_analysis.py        # Análisis de ocupación
    ├── delay_analysis.py            # Análisis de retrasos
    └── visualization.py             # Utilidades de visualización
```

### Requisitos del Sistema

- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows, macOS, Linux
- **Memoria RAM**: Mínimo 2GB (4GB recomendado)
- **Espacio en disco**: Mínimo 500MB

### Instalación

#### 1. Clonar o descargar el repositorio

```bash
git clone <url-del-repositorio>
cd DataProj
```

#### 2. Crear un entorno virtual (recomendado)

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### Uso

#### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en su navegador web en `http://localhost:8501`

#### Instrucciones de uso

1. **Cargar datos:**
   - En la barra lateral, seleccione "Cargar archivo CSV"
   - Cargue un archivo CSV con el formato especificado
   - El sistema validará automáticamente la estructura

2. **Filtrar datos:**
   - Configure los filtros deseados en la barra lateral
   - Los gráficos se actualizarán automáticamente

3. **Explorar análisis:**
   - Navegue entre las pestañas disponibles
   - Interactúe con los gráficos (zoom, desplazamiento, etc.)
   - Consulte las tablas de datos detalladas

4. **Descargar resultados:**
   - Use los botones de descarga para obtener datos en CSV
   - Los gráficos se pueden guardar como imágenes (Plotly)

### Formato del Archivo CSV

El archivo CSV debe contener las siguientes columnas:

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| fecha | DATE | Fecha del servicio (YYYY-MM-DD) | 2024-01-01 |
| hora | INTEGER | Hora del servicio (0-23) | 7 |
| linea | STRING | Identificador de línea de transporte | L1, L2, L3 |
| zona | STRING | Zona geográfica | Centro, Norte, Sur, Este, Oeste |
| pasajeros | INTEGER | Número de pasajeros | 120 |
| ocupacion | FLOAT | Porcentaje de ocupación (0-100) | 75.5 |
| retraso | FLOAT | Retraso en minutos | 5.2 |

#### Ejemplo de archivo CSV:

```csv
fecha,hora,linea,zona,pasajeros,ocupacion,retraso
2024-01-01,6,L1,Centro,45,35,2
2024-01-01,7,L1,Centro,120,65,5
2024-01-01,8,L1,Centro,180,85,12
```

Un archivo de ejemplo (`sample_data.csv`) está incluido en el proyecto para pruebas.

### Descripción de Variables Derivadas

El sistema crea automáticamente las siguientes variables durante el procesamiento:

- **dia_semana**: Día de la semana (0=Lunes, 6=Domingo)
- **nombre_dia**: Nombre del día en inglés
- **franja_horaria**: Categorización horaria
  - Madrugada: 0-6 horas
  - Mañana: 6-12 horas
  - Tarde: 12-18 horas
  - Noche: 18-24 horas
- **es_fin_semana**: 1 si es fin de semana, 0 en caso contrario
- **tipo_dia**: "Fin de semana" o "Entre semana"
- **es_puntual**: 1 si retraso < 5 min, 0 en caso contrario
- **severidad_retraso**: Categoría de retraso (Puntual, Leve, Moderado, Severo)
- **nivel_ocupacion**: Nivel de ocupación (Bajo, Medio, Alto, Muy Alto)

### Guía de Filtros

Los filtros disponibles en la barra lateral permiten:

- **Rango de Fechas**: Seleccione período a analizar
- **Línea de Transporte**: Filtre por líneas específicas
- **Zona Geográfica**: Seleccione zonas de interés
- **Día de la Semana**: Filtre por días específicos
- **Franja Horaria**: Seleccione períodos del día

Todos los gráficos se actualizan automáticamente según los filtros activos.

### Indicadores Clave (KPIs)

**Dashboard Principal muestra:**

1. **Total de Pasajeros**: Suma de todos los pasajeros en el período
2. **Ocupación Promedio Global (%)**: Media de ocupación de todos los servicios
3. **Retraso Promedio (minutos)**: Media de retrasos en minutos
4. **Tasa de Puntualidad (%)**: Porcentaje de servicios con retraso < 5 minutos

### Módulos del Sistema

#### `data_loader.py`
Maneja la carga y validación de archivos CSV.
- `load_csv_file()`: Carga archivo CSV
- `validate_csv_structure()`: Valida columnas requeridas
- `get_data_info()`: Obtiene información del dataset

#### `preprocessing.py`
Preprocesa y transforma los datos.
- `clean_data()`: Limpia valores faltantes y anomalías
- `transform_datetime_features()`: Convierte campos de fecha/hora
- `create_derived_metrics()`: Crea variables derivadas
- `preprocess_full_pipeline()`: Ejecuta pipeline completo

#### `flow_analysis.py`
Analiza el flujo de pasajeros.
- `get_hourly_passengers()`: Pasajeros por hora
- `get_daily_passengers()`: Pasajeros por día
- `get_zone_passengers()`: Pasajeros por zona
- `get_peak_hours()`: Identifica horas pico

#### `occupancy_analysis.py`
Analiza la ocupación del sistema.
- `get_occupancy_by_line()`: Ocupación por línea
- `get_occupancy_by_zone()`: Ocupación por zona
- `get_occupancy_heatmap()`: Matriz de ocupación
- `get_critical_lines()`: Identifica líneas críticas

#### `delay_analysis.py`
Analiza los retrasos del sistema.
- `get_delay_statistics()`: Estadísticas generales
- `get_delay_by_line()`: Retraso por línea
- `get_delay_by_zone()`: Retraso por zona
- `get_punctuality_rate()`: Calcula tasa de puntualidad

#### `visualization.py`
Utilidades para crear visualizaciones.
- `create_line_chart()`: Gráfico de líneas
- `create_bar_chart()`: Gráfico de barras
- `create_histogram()`: Histograma
- `create_scatter_plot()`: Diagrama de dispersión
- `create_heatmap()`: Mapa de calor

### Características Técnicas

**Interfaz responsiva**: Optimizada para diferentes tamaños de pantalla
**Gráficos interactivos**: Todos los gráficos son interactivos con Plotly
**Filtros dinámicos**: Los datos se actualizan en tiempo real
**Validación de datos**: Manejo robusto de datos incompletos
**Exportación flexible**: Descarga de datos en múltiples formatos
**Interfaz en español**: Todos los textos visibles están en español
**Código modular**: Fácil mantenimiento y extensión

### Solución de Problemas

**Error: "ModuleNotFoundError: No module named 'streamlit'"**
```bash
pip install -r requirements.txt
```

**Error: "No hay datos que coincidan con los filtros"**
- Verifique que el archivo CSV sea correcto
- Verifique que los filtros sean demasiado restrictivos
- Cargue el archivo sample_data.csv para probar

**La aplicación es lenta**
- Reduce el rango de fechas
- Filtra por líneas o zonas específicas
- Asegúrate de tener suficiente RAM disponible

**Los gráficos no se muestran**
- Verifica que Plotly está instalado: `pip install plotly`
- Recarga la página del navegador

### Extensiones Futuras

Posibles mejoras y extensiones:

- Integración con mapas interactivos (folium)
- Análisis predictivo con machine learning
- Modelos de forecasting
- Sistema de alertas automáticas
- Informes automáticos en PDF
- API REST para integración con otros sistemas
- Sistema de autenticación de usuarios
- Versión mobile
- Soporte multiidioma

### Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

### Autor

Desarrollado como sistema académico para análisis y visualización de transporte urbano.

**Universidad**: ULEAM (Universidad Laica Eloy Alfaro de Manabí)
**Curso**: Visualización de Datos
**Período**: 2024

### Soporte

Para reportar problemas o sugerencias:
- Abre un issue en el repositorio
- Envía un email con detalles del problema
- Consulta la documentación en línea

### Agradecimientos

- Streamlit por la excelente framework de visualización
- Plotly por los gráficos interactivos
- Pandas por el manejo de datos
- Comunidad de Python por las librerías utilizadas

---

**Versión**: 1.0.0  
**Última actualización**: Enero 2024  
**Estado**: Producción 
