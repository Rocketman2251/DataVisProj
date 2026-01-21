# Page Configuration
PAGE_TITLE = "Sistema de Transporte Urbano"
PAGE_ICON = "img/logo.ico"  # Custom icon for browser tab
PAGE_LAYOUT = "wide"

# Streamlit Native Icons (Material Symbols)
# Syntax: :material/icon_name:
ICON_DASHBOARD = ":material/dashboard:"
ICON_SETTINGS = ":material/settings:"
ICON_UPLOAD = ":material/cloud_upload:"
ICON_FILTER = ":material/filter_alt:"
ICON_STATS = ":material/analytics:"
ICON_GROUP = ":material/groups:"
ICON_BUS = ":material/directions_bus:"
ICON_TIME = ":material/schedule:"
ICON_WARNING = ":material/warning:"
ICON_CHECK = ":material/check_circle:"
ICON_ERROR = ":material/error:"
ICON_INFO = ":material/info:"
ICON_DOWNLOAD = ":material/download:"
ICON_MAP = ":material/map:"

# Colors
COLOR_PRIMARY = "#1f77b4"
COLOR_BACKGROUND = "#f0f2f6"
COLOR_ACCENT = "#FF6B35"

# Analysis Thresholds
THRESHOLD_CRITICAL_OCCUPANCY = 75.0  # Percentage
THRESHOLD_PUNCTUALITY = 5.0          # Minutes

# Custom CSS
CUSTOM_CSS = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    .main-title {
        text-align: center;
        color: #1E3D59;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .landing-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    .landing-header {
        color: #1E3D59;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .step-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1rem;
        gap: 15px;
    }
    .step-icon {
        background-color: #e3f2fd;
        color: #1976d2;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        flex-shrink: 0;
    }
    .step-content {
        color: #444;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
"""

# HTML Template for Landing Page
LANDING_PAGE_HTML = """
<div class="landing-card">
<div class="landing-header">
<i class="fas fa-bus-alt"></i> Bienvenido al Sistema de Análisis de Transporte
</div>
<p style="color: #666; margin-bottom: 1.5rem;">
Este sistema avanzado permite el análisis integral de datos de transporte público.
Siga estos pasos para comenzar:
</p>
<div class="step-item">
<div class="step-icon">1</div>
<div class="step-content">
<strong>Cargue sus datos</strong><br>
Use el panel lateral para subir su archivo CSV.
</div>
</div>
<div class="step-item">
<div class="step-icon">2</div>
<div class="step-content">
<strong>Configure filtros</strong><br>
Seleccione rangos de fechas, líneas y zonas específicas.
</div>
</div>
<div class="step-item">
<div class="step-icon"><i class="fas fa-chart-line"></i></div>
<div class="step-content">
<strong>Visualice resultados</strong><br>
Explore los dashboards interactivos y KPIs en tiempo real.
</div>
</div>
<div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #dee2e6;">
<small style="color: #888;">
<i class="fas fa-info-circle"></i> 
Asegúrese de que su archivo CSV cumpla con el formato requerido (fecha, hora, linea, zona, etc.)
</small>
</div>
</div>
"""
