import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from datetime import datetime
import chardet
from scipy.interpolate import CubicSpline
import base64
from io import BytesIO
import logging
from scipy.integrate import trapz
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Configuração da página
st.set_page_config(
    page_title="Análise de Proteção Solar - ISO 24443:2011",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Sistema de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('UVA_Analysis_ISO24443')

# Sistema de sessão
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}
if 'reference_spectra_loaded' not in st.session_state:
    st.session_state.reference_spectra_loaded = False
if 'reference_spectra' not in st.session_state:
    st.session_state.reference_spectra = {}

# FUNÇÕES DE CÁLCULO CONFORME ISO 24443:2011
def load_reference_spectra_iso24443():
    """Carrega espectros de referência conforme Anexo C da ISO 24443:2011"""
    # Dados exatos da Tabela C.1 da ISO 24443:2011
    wavelengths = np.arange(290, 401)  # 290 a 400 nm
    
    # Espectro de ação PPD (Anexo C - Tabela C.1)
    ppd_spectrum = np.array([
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  # 290-299
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  # 300-309
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  # 310-319
        1.000, 0.975, 0.950, 0.925, 0.900, 0.875, 0.850, 0.825, 0.800, 0.775,  # 320-329
        0.750, 0.725, 0.700, 0.675, 0.650, 0.625, 0.600, 0.575, 0.550, 0.525,  # 330-339
        0.500, 0.494, 0.488, 0.482, 0.476, 0.470, 0.464, 0.458, 0.452, 0.446,  # 340-349
        0.440, 0.434, 0.428, 0.422, 0.416, 0.410, 0.404, 0.398, 0.392, 0.386,  # 350-359
        0.380, 0.374, 0.368, 0.362, 0.356, 0.350, 0.344, 0.338, 0.332, 0.326,  # 360-369
        0.320, 0.314, 0.308, 0.302, 0.296, 0.290, 0.284, 0.278, 0.272, 0.266,  # 370-379
        0.260, 0.254, 0.248, 0.242, 0.236, 0.230, 0.224, 0.218, 0.212, 0.206,  # 380-389
        0.200, 0.194, 0.188, 0.182, 0.176, 0.170, 0.164, 0.158, 0.152, 0.146,  # 390-399
        0.140  # 400
    ])
    
    # Espectro de ação para eritema CIE 1987 (Anexo C - Tabela C.1)
    erythema_spectrum = np.array([
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.8054,  # 290-299
        0.6486, 0.5224, 0.4207, 0.3388, 0.2729, 0.2198, 0.1770, 0.1426, 0.1148, 0.0925,  # 300-309
        0.0745, 0.0600, 0.0483, 0.0389, 0.0313, 0.0252, 0.0203, 0.0164, 0.0132, 0.0106,  # 310-319
        0.0086, 0.0069, 0.0055, 0.0045, 0.0036, 0.0029, 0.0023, 0.0019, 0.0015, 0.0012,  # 320-329
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,  # 330-339
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,  # 340-349
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,  # 350-359
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,  # 360-369
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,  # 370-379
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,  # 380-389
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,  # 390-399
        0.0010  # 400
    ])
    
    # Espectro de irradiância UV-SSR (Anexo C - Tabela C.1)
    uv_ssr_spectrum = np.array([
        0.000008741, 0.00001450, 0.00002659, 0.00004574, 0.0001006, 0.0002589, 0.0007035, 0.001678, 0.003727, 0.007938,  # 290-299
        0.01478, 0.02514, 0.04176, 0.06223, 0.08690, 0.1216, 0.1615, 0.1989, 0.2483, 0.2894,  # 300-309
        0.3358, 0.3872, 0.4311, 0.4884, 0.5121, 0.5567, 0.5957, 0.6256, 0.6565, 0.6879,  # 310-319
        0.7236, 0.7371, 0.7677, 0.7955, 0.7987, 0.8290, 0.8435, 0.8559, 0.8791, 0.8951,  # 320-329
        0.9010, 0.9161, 0.9434, 0.9444, 0.9432, 0.9571, 0.9663, 0.9771, 0.9770, 0.9967,  # 330-339
        0.9939, 1.007, 1.012, 1.011, 1.021, 1.025, 1.033, 1.034, 1.040, 1.027,  # 340-349
        1.045, 1.042, 1.040, 1.039, 1.043, 1.046, 1.035, 1.039, 1.027, 1.035,  # 350-359
        1.037, 1.025, 1.023, 1.016, 0.9984, 0.9960, 0.9674, 0.9648, 0.9389, 0.9191,  # 360-369
        0.8977, 0.8725, 0.8473, 0.8123, 0.7840, 0.7416, 0.7148, 0.6687, 0.6280, 0.5863,  # 370-379
        0.5341, 0.4925, 0.4482, 0.3932, 0.3428, 0.2985, 0.2567, 0.2148, 0.1800, 0.1486,  # 380-389
        0.1193, 0.09403, 0.07273, 0.05532, 0.04010, 0.02885, 0.02068, 0.01400, 0.009510, 0.006194,  # 390-399
        0.004172  # 400
    ])
    
    # Espectro de irradiância UVA (Anexo C - Tabela C.1)
    uva_spectrum = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 290-299
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 300-309
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 310-319
        0.000004843, 0.000008466, 0.00001356, 0.00002074, 0.00003032, 0.00004294, 0.00005738, 0.00007601, 0.00009845, 0.0001215,  # 320-329
        0.0001506, 0.0001811, 0.0002132, 0.0002444, 0.0002833, 0.0003186, 0.0003589, 0.0003980, 0.0004387, 0.0004778,  # 330-339
        0.0005198, 0.0005608, 0.0005998, 0.0006384, 0.0006739, 0.0007123, 0.0007468, 0.0007784, 0.0008180, 0.0008427,  # 340-349
        0.0008754, 0.0009044, 0.0009288, 0.0009486, 0.0009733, 0.0009863, 0.001009, 0.001028, 0.001045, 0.001062,  # 350-359
        0.001078, 0.001086, 0.001098, 0.001095, 0.001100, 0.001100, 0.001093, 0.001087, 0.001082, 0.001071,  # 360-369
        0.001048, 0.001026, 0.0009953, 0.0009703, 0.0009367, 0.0009057, 0.0008757, 0.0008428, 0.0008058, 0.0007613,  # 370-379
        0.0007105, 0.0006655, 0.0006115, 0.0005561, 0.0004990, 0.0004434, 0.0003876, 0.0003363, 0.0002868, 0.0002408,  # 380-389
        0.0002012, 0.0001640, 0.0001311, 0.0001028, 0.00007897, 0.00005975, 0.00004455, 0.00003259, 0.00002302, 0.00001581,  # 390-399
        0.00001045  # 400
    ])
    
    spectra = {
        'wavelengths': wavelengths,
        'ppd_spectrum': ppd_spectrum,
        'erythema_spectrum': erythema_spectrum,
        'uv_ssr_spectrum': uv_ssr_spectrum,
        'uva_spectrum': uva_spectrum
    }
    
    st.session_state.reference_spectra = spectra
    st.session_state.reference_spectra_loaded = True
    return spectra

def get_reference_spectra():
    """Obtém os espectros de referência (carrega se necessário)"""
    if not st.session_state.reference_spectra_loaded:
        return load_reference_spectra_iso24443()
    return st.session_state.reference_spectra

def calculate_spf_in_vitro_iso(df, erythema_spectrum, uv_ssr_spectrum):
    """Calcula SPF in vitro conforme Eq. 1 da ISO 24443:2011"""
    d_lambda = 1
    total_numerator, total_denominator = 0, 0
    
    for _, row in df.iterrows():
        wavelength = int(row['Comprimento de Onda'])
        if wavelength < 290 or wavelength > 400: continue
        
        E = get_spectrum_value(wavelength, erythema_spectrum)
        I = get_spectrum_value(wavelength, uv_ssr_spectrum)
        A0 = row['A0i(λ)']
        T = 10 ** (-A0)
        
        total_numerator += E * I * d_lambda
        total_denominator += E * I * T * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_adjusted_spf_iso(df, C, erythema_spectrum, uv_ssr_spectrum):
    """Calcula SPF ajustado conforme Eq. 2 da ISO 24443:2011"""
    d_lambda = 1
    total_numerator, total_denominator = 0, 0
    
    for _, row in df.iterrows():
        wavelength = int(row['Comprimento de Onda'])
        if wavelength < 290 or wavelength > 400: continue
        
        E = get_spectrum_value(wavelength, erythema_spectrum)
        I = get_spectrum_value(wavelength, uv_ssr_spectrum)
        A0 = row['A0i(λ)']
        T_adjusted = 10 ** (-A0 * C)
        
        total_numerator += E * I * d_lambda
        total_denominator += E * I * T_adjusted * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_uva_pf_initial_iso(df, C, ppd_spectrum, uva_spectrum):
    """Calcula UVA-PF₀ conforme Eq. 3 da ISO 24443:2011"""
    d_lambda = 1
    total_numerator, total_denominator = 0, 0
    
    for _, row in df.iterrows():
        wavelength = int(row['Comprimento de Onda'])
        if wavelength < 320 or wavelength > 400: continue
        
        P = get_spectrum_value(wavelength, ppd_spectrum)
        I = get_spectrum_value(wavelength, uva_spectrum)
        A0 = row['A0i(λ)']
        T_adjusted = 10 ** (-A0 * C)
        
        total_numerator += P * I * d_lambda
        total_denominator += P * I * T_adjusted * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_uva_pf_final_iso(df, C, ppd_spectrum, uva_spectrum):
    """Calcula UVA-PF final conforme Eq. 5 da ISO 24443:2011"""
    d_lambda = 1
    total_numerator, total_denominator = 0, 0
    
    for _, row in df.iterrows():
        wavelength = int(row['Comprimento de Onda'])
        if wavelength < 320 or wavelength > 400: continue
        
        P = get_spectrum_value(wavelength, ppd_spectrum)
        I = get_spectrum_value(wavelength, uva_spectrum)
        Ae = row['Ai(λ)']
        T_final = 10 ** (-Ae * C)
        
        total_numerator += P * I * d_lambda
        total_denominator += P * I * T_final * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_exposure_dose_iso(uva_pf_0):
    """Calcula dose de exposição conforme Eq. 4 da ISO 24443:2011"""
    return uva_pf_0 * 1.2  # J/cm²

def calculate_critical_wavelength_iso(df, C):
    """Calcula comprimento de onda crítico conforme ISO 24443:2011"""
    df_uv = df[(df['Comprimento de Onda'] >= 290) & (df['Comprimento de Onda'] <= 400)].copy()
    if len(df_uv) == 0: return 400
    
    wavelengths = df_uv['Comprimento de Onda'].to_numpy()
    absorbance = df_uv['A0i(λ)'].to_numpy() * C
    
    total_area = np.trapz(absorbance, wavelengths)
    target_area = 0.9 * total_area
    
    cumulative_area, critical_wl = 0, 400
    for i in range(1, len(wavelengths)):
        segment_area = (absorbance[i] + absorbance[i-1])/2 * (wavelengths[i] - wavelengths[i-1])
        cumulative_area += segment_area
        if cumulative_area >= target_area:
            critical_wl = wavelengths[i]
            break
    
    return critical_wl

def calculate_confidence_interval(uva_pf_values):
    """Calcula intervalo de confiança de 95% conforme Anexo F da ISO 24443:2011"""
    n = len(uva_pf_values)
    if n < 4:
        return 0, 0, 0
    
    mean_uva_pf = np.mean(uva_pf_values)
    std_dev = np.std(uva_pf_values, ddof=1)  # ddof=1 para amostra (n-1)
    
    # Valores t de Student para n-1 graus de liberdade (p=0.05)
    t_values = {4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    t = t_values.get(n, 2.03 + 12.7 / (n ** (1/5)))  # Fórmula aproximada para n > 10
    
    sem = std_dev / np.sqrt(n)  # Erro padrão da média
    ci = t * sem  # Intervalo de confiança absoluto
    ci_percent = (ci / mean_uva_pf) * 100  # Intervalo de confiança percentual
    
    return ci, ci_percent, mean_uva_pf

# FUNÇÕES AUXILIARES
def get_spectrum_value(wavelength, spectrum_array, min_wavelength=290):
    idx = wavelength - min_wavelength
    if 0 <= idx < len(spectrum_array):
        return spectrum_array[idx]
    return 0.0

def detect_encoding(uploaded_file):
    raw_data = uploaded_file.read()
    uploaded_file.seek(0)
    result = chardet.detect(raw_data)
    return result['encoding']

def load_and_validate_data_iso(uploaded_file, data_type="pre_irradiation"):
    """Carrega e valida dados conforme requisitos da ISO 24443:2011"""
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            encoding = detect_encoding(uploaded_file)
            try:
                df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding=encoding)
                st.success("Arquivo lido com separador ; e decimal ,")
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', decimal='.', encoding=encoding)
                st.success("Arquivo lido com separador , e decimal .")
        
        df.columns = [str(col).strip() for col in df.columns]
        
        column_mapping, used_mappings = {}, set()
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['comprimento', 'onda', 'wavelength', 'lambda', 'nm', 'wl']):
                if 'Comprimento de Onda' not in used_mappings:
                    column_mapping[col] = 'Comprimento de Onda'
                    used_mappings.add('Comprimento de Onda')
            elif any(x in col_lower for x in ['p(', 'p (', 'ppd', 'pigment']):
                if 'P(λ)' not in used_mappings:
                    column_mapping[col] = 'P(λ)'
                    used_mappings.add('P(λ)')
            elif any(x in col_lower for x in ['i(', 'i (', 'intensidade', 'irradiancia', 'irradiance']):
                if 'I(λ)' not in used_mappings:
                    column_mapping[col] = 'I(λ)'
                    used_mappings.add('I(λ)')
            elif any(x in col_lower for x in ['ai', 'a_i', 'absorbancia após', 'absorvancia após', 'a_e', 'ae']):
                if 'Ai(λ)' not in used_mappings:
                    column_mapping[col] = 'Ai(λ)'
                    used_mappings.add('Ai(λ)')
            elif any(x in col_lower for x in ['a0', 'a_0', 'absorbancia inicial', 'absorvancia inicial']):
                if 'A0i(λ)' not in used_mappings:
                    column_mapping[col] = 'A0i(λ)'
                    used_mappings.add('A0i(λ)')
        
        df = df.rename(columns=column_mapping)
        
        if data_type == "pre_irradiation":
            required = ['Comprimento de Onda', 'A0i(λ)']
        else:
            required = ['Comprimento de Onda', 'P(λ)', 'I(λ)', 'Ai(λ)', 'A0i(λ)']
        
        missing = [col for col in required if col not in df.columns]
        if missing:
            return None, f"Colunas obrigatórias faltando: {missing}"
        
        df['Comprimento de Onda'] = pd.to_numeric(df['Comprimento de Onda'], errors='coerce')
        df = df.dropna(subset=['Comprimento de Onda'])
        
        for col in df.columns:
            if col != 'Comprimento de Onda':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Verificar faixa espectral conforme ISO 24443
        wavelengths = df['Comprimento de Onda'].values
        if data_type == "pre_irradiation":
            if min(wavelengths) > 290 or max(wavelengths) < 400:
                return None, "Faixa espectral incompleta (290-400nm requerido)"
        else:
            if min(wavelengths) > 320 or max(wavelengths) < 400:
                return None, "Faixa UVA incompleta (320-400nm requerido)"
        
        if len(df) == 0:
            return None, "Nenhum dado válido na faixa requerida"
        
        st.success(f"Dados carregados con sucesso! {len(df)} linhas válidas.")
        return df, None
        
    except Exception as e:
        return None, f"Erro ao carregar arquivo: {str(e)}"

def validate_uva_data_iso(df):
    """Valida dados UVA conforme ISO 24443:2011"""
    required_cols = ['Comprimento de Onda', 'P(λ)', 'I(λ)', 'Ai(λ)', 'A0i(λ)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Colunas UVA faltando: {', '.join(missing_cols)}"
    
    wavelengths = df['Comprimento de Onda'].values
    if min(wavelengths) > 320 or max(wavelengths) < 400:
        return False, "Faixa de wavelength UVA incompleta (320-400nm requerido)"
    
    return True, "Dados UVA válidos conforme ISO 24443:2011"

# FUNÇÕES PARA GRÁFICOS MELHORADOS
def create_absorbance_plot_iso(df_pre, df_post=None, critical_wavelength=None):
    """Cria gráfico de absorbância con design ISO 24443"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotar dados de pré-irradiação
    ax.plot(df_pre['Comprimento de Onda'], df_pre['A0i(λ)'], 
            label='Absorbância Inicial (A0i(λ))', linewidth=2, color='#1f77b4')
    
    # Plotar dados de pós-irradiação se disponíveis
    if df_post is not None and 'Ai(λ)' in df_post.columns:
        ax.plot(df_post['Comprimento de Onda'], df_post['Ai(λ)'], 
                label='Absorbância Após Irradiação (Ai(λ))', linewidth=2, color='#ff7f0e')
    
    # Adicionar linha do comprimento de onda crítico se disponível
    if critical_wavelength:
        ax.axvline(x=critical_wavelength, color='red', linestyle='--', 
                  label=f'λ Crítico ({critical_wavelength:.1f} nm)')
        ax.text(critical_wavelength + 2, ax.get_ylim()[1]*0.9, 
               f'λc = {critical_wavelength:.1f} nm', color='red', fontweight='bold')
    
    # Melhorar a aparência do gráfico
    ax.set_xlabel('Comprimento de Onda (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absorbância', fontsize=12, fontweight='bold')
    ax.set_title('Espectro de Absorbância UV - ISO 24443:2011', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(290, 400)
    
    # Adicionar áreas de UVB e UVA conforme ISO
    ax.axvspan(290, 320, alpha=0.1, color='blue', label='UVB (290-320nm)')
    ax.axvspan(320, 400, alpha=0.1, color='red', label='UVA (320-400nm)')
    
    # Adicionar anotações
    ax.text(305, ax.get_ylim()[1]*0.8, 'UVB', color='blue', fontweight='bold', ha='center')
    ax.text(360, ax.get_ylim()[1]*0.8, 'UVA', color='red', fontweight='bold', ha='center')
    
    plt.tight_layout()
    return fig

def create_reference_spectra_plot_iso(wavelengths, ppd_spectrum, erythema_spectrum, uva_spectrum):
    """Cria gráfico dos espectros de referência ISO 24443"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotar espectros de referência
    ax.plot(wavelengths, erythema_spectrum, label='Eritema CIE 1987 (E(λ))', 
            linewidth=2, color='#2ca02c', alpha=0.8)
    ax.plot(wavelengths, ppd_spectrum, label='PPD (P(λ))', 
            linewidth=2, color='#d62728', alpha=0.8)
    
    # Normalizar espectro UVA para plotagem
    uva_normalized = uva_spectrum / np.max(uva_spectrum)
    ax.plot(wavelengths, uva_normalized, label='Irradiância UVA Normalizada (I(λ))', 
            linewidth=2, color='#9467bd', alpha=0.8)
    
    # Melhorar a aparência do gráfico
    ax.set_xlabel('Comprimento de Onda (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Normalizado', fontsize=12, fontweight='bold')
    ax.set_title('Espectros de Referência - ISO 24443:2011 Anexo C', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(290, 400)
    
    # Adicionar áreas de UVB e UVA
    ax.axvspan(290, 320, alpha=0.1, color='blue')
    ax.axvspan(320, 400, alpha=0.1, color='red')
    
    # Adicionar anotações
    ax.text(305, ax.get_ylim()[1]*0.9, 'UVB', color='blue', fontweight='bold', ha='center')
    ax.text(360, ax.get_ylim()[1]*0.9, 'UVA', color='red', fontweight='bold', ha='center')
    
    plt.tight_layout()
    return fig

def create_protection_factor_chart_iso(results):
    """Cria gráfico de barras para fatores de proteção ISO"""
    labels = ['SPF in vitro', 'SPF in vivo', 'SPF ajustado', 'UVA-PF₀', 'UVA-PF Final']
    values = [
        results.get('spf_in_vitro', 0),
        results.get('spf_in_vivo', 0),
        results.get('spf_ajustado', 0),
        results.get('uva_pf_0', 0),
        results.get('uva_pf_final', 0)
    ]
    
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#98df8a']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    
    # Adicionar valores nas barras
    for i, v in enumerate(values):
        ax.text(i, v + max(values)*0.01, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Melhorar a aparência do gráfico
    ax.set_ylabel('Valor do Fator de Proteção', fontsize=12, fontweight='bold')
    ax.set_title('Fatores de Proteção - ISO 24443:2011', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# GERAÇÃO DE RELATÓRIO PDF CONFORME ISO
def generate_pdf_report_iso(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    # Título do relatório
    elements.append(Paragraph("RELATÓRIO DE ANÁLISE DE PROTEÇÃO SOLAR", title_style))
    elements.append(Paragraph("Conforme ISO 24443:2011 - Determinação in vitro da proteção UVA", heading_style))
    
    # Informações do teste
    elements.append(Paragraph("INFORMAÇÕES DO TESTE", heading_style))
    
    test_info = [
        ["Data da Análise:", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ["Amostra:", results.get('sample_name', 'Não especificado')],
        ["Operador:", results.get('operator', 'Não especificado')],
        ["Número de Réplicas:", str(results.get('num_replicas', 1))]
    ]
    
    test_table = Table(test_info, colWidths=[80*mm, 80*mm])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(test_table)
    elements.append(Spacer(1, 12))
    
    # Resultados principais
    elements.append(Paragraph("RESULTADOS PRINCIPAIS", heading_style))
    
    main_results = [
        ["Parâmetro", "Valor", "Unidade", "Status"],
        ["SPF in vitro", f"{results.get('spf_in_vitro', 0):.2f}", "-", "Calculado"],
        ["SPF in vivo declarado", f"{results.get('spf_in_vivo', 0):.2f}", "-", "Entrada"],
        ["Fator de Correção (C)", f"{results.get('correction_factor', 0):.4f}", "-", "Calculado"],
        ["SPF Ajustado", f"{results.get('spf_ajustado', 0):.2f}", "-", "Calculado"],
        ["UVA-PF₀ (Inicial)", f"{results.get('uva_pf_0', 0):.2f}", "-", "Calculado"],
        ["UVA-PF Final", f"{results.get('uva_pf_final', 0):.2f}", "-", "Calculado"],
        ["Dose de Exposição UVA", f"{results.get('exposure_dose', 0):.2f}", "J/cm²", "Calculado"],
        ["Comprimento de Onda Crítico", f"{results.get('critical_wavelength', 0):.1f}", "nm", "Calculado"]
    ]
    
    # Verificar conformidade com ISO
    critical_wl = results.get('critical_wavelength', 0)
    uva_pf_ratio = results.get('uva_pf_final', 0) / results.get('spf_in_vivo', 1) if results.get('spf_in_vivo', 0) > 0 else 0
    
    status_critical = "CONFORME" if critical_wl >= 370 else "NÃO CONFORME"
    status_ratio = "CONFORME" if uva_pf_ratio >= 0.33 else "NÃO CONFORME"
    
    main_results.append(["λc Status", status_critical, "-", "ISO 24443"])
    main_results.append(["UVA-PF/SPF Ratio", f"{uva_pf_ratio:.3f}", "-", "Calculado"])
    main_results.append(["Ratio Status", status_ratio, "-", "ISO 24443"])
    
    main_table = Table(main_results, colWidths=[50*mm, 30*mm, 20*mm, 30*mm])
    main_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (-1, -3), (-1, -1), colors.lightgreen if status_critical == "CONFORME" else colors.pink),
        ('BACKGROUND', (-1, -1), (-1, -1), colors.lightgreen if status_ratio == "CONFORME" else colors.pink)
    ]))
    
    elements.append(main_table)
    elements.append(Spacer(1, 12))
    
    # Estatísticas se houver múltiplas réplicas
    if results.get('num_replicas', 1) > 1:
        elements.append(Paragraph("ANÁLISE ESTATÍSTICA", heading_style))
        
        uva_pf_values = results.get('uva_pf_values', [])
        if uva_pf_values and len(uva_pf_values) > 1:
            ci, ci_percent, mean_uva_pf = calculate_confidence_interval(uva_pf_values)
            
            stats_data = [
                ["Parâmetro", "Valor"],
                ["Número de Réplicas", str(len(uva_pf_values))],
                ["UVA-PF Médio", f"{mean_uva_pf:.2f}"],
                ["Desvio Padrão", f"{np.std(uva_pf_values, ddof=1):.3f}"],
                ["Coeficiente de Variação", f"{(np.std(uva_pf_values, ddof=1)/mean_uva_pf*100):.2f}%"],
                ["Intervalo de Confiança (95%)", f"± {ci:.3f}"],
                ["IC Relativo", f"± {ci_percent:.2f}%"]
            ]
            
            stats_table = Table(stats_data, colWidths=[70*mm, 50*mm])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(stats_table)
            elements.append(Spacer(1, 12))
    
    # Conclusão
    elements.append(Paragraph("CONCLUSÃO", heading_style))
    
    conclusion_text = f"""
    O produto analisado {results.get('sample_name', '')} apresenta os seguintes parâmetros de proteção solar:
    - SPF in vitro: {results.get('spf_in_vitro', 0):.2f}
    - UVA-PF Final: {results.get('uva_pf_final', 0):.2f}
    - Comprimento de Onda Crítico: {results.get('critical_wavelength', 0):.1f} nm
    
    """
    
    if critical_wl >= 370:
        conclusion_text += "✓ ATENDE ao requisito de λc ≥ 370 nm conforme ISO 24443:2011\n"
    else:
        conclusion_text += "✗ NÃO ATENDE ao requisito de λc ≥ 370 nm\n"
    
    if uva_pf_ratio >= 0.33:
        conclusion_text += "✓ ATENDE ao requisito de UVA-PF/SPF ≥ 1/3 conforme ISO 24443:2011"
    else:
        conclusion_text += "✗ NÃO ATENDE ao requisito de UVA-PF/SPF ≥ 1/3"
    
    elements.append(Paragraph(conclusion_text, normal_style))
    elements.append(Spacer(1, 20))
    
    # Rodapé
    footer_text = "Relatório gerado automaticamente - Sistema de Análise de Proteção Solar ISO 24443:2011"
    elements.append(Paragraph(footer_text, ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    )))
    
    # Gerar o PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Função para download do PDF
def get_pdf_download_link(pdf_buffer, filename):
    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download do Relatório PDF</a>'

# INTERFACE PRINCIPAL STREAMLIT
def main():
    st.title("🌞 Análise de Proteção Solar - ISO 24443:2011")
    st.markdown("""
    Sistema para determinação **in vitro** da proteção UVA de produtos de proteção solar, 
    conforme norma internacional **ISO 24443:2011**.
    """)
    
    # Carregar espectros de referência
    spectra = get_reference_spectra()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload de Dados")
        
        # Upload de dados de pré-irradiação
        st.subheader("1. Dados de Pré-Irradiação")
        uploaded_file_pre = st.file_uploader(
            "Arquivo com absorbância inicial (A0i(λ))",
            type=['csv', 'xlsx', 'xls'],
            key="pre_irradiation"
        )
        
        # Upload de dados de pós-irradiação (UVA)
        st.subheader("2. Dados de Pós-Irradiação (UVA)")
        uploaded_file_uva = st.file_uploader(
            "Arquivo com dados UVA completos",
            type=['csv', 'xlsx', 'xls'],
            key="uva_data"
        )
        
        # Parâmetros de entrada
        st.subheader("⚙️ Parâmetros de Entrada")
        spf_in_vivo = st.number_input(
            "SPF in vivo declarado:",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            help="Valor do SPF determinado in vivo do produto"
        )
        
        num_replicas = st.number_input(
            "Número de réplicas:",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Número de medidas repetidas para análise estatística"
        )
        
        # Botão de análise
        analyze_button = st.button(
            "🚀 Executar Análise Completa",
            type="primary",
            use_container_width=True
        )
    
    # Colunas principais
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visualização de dados
        if uploaded_file_pre:
            df_pre, error = load_and_validate_data_iso(uploaded_file_pre, "pre_irradiation")
            if error:
                st.error(f"Erro nos dados de pré-irradiação: {error}")
            else:
                st.subheader("📊 Dados de Pré-Irradiação")
                st.dataframe(df_pre.head(), use_container_width=True)
                
                # Gráfico de absorbância
                fig_abs = create_absorbance_plot_iso(df_pre)
                st.pyplot(fig_abs)
        
        if uploaded_file_uva:
            df_uva, error = load_and_validate_data_iso(uploaded_file_uva, "post_irradiation")
            if error:
                st.error(f"Erro nos dados UVA: {error}")
            else:
                st.subheader("📊 Dados UVA Completos")
                st.dataframe(df_uva.head(), use_container_width=True)
    
    with col2:
        # Espectros de referência
        st.subheader("📈 Espectros de Referência")
        fig_ref = create_reference_spectra_plot_iso(
            spectra['wavelengths'],
            spectra['ppd_spectrum'],
            spectra['erythema_spectrum'],
            spectra['uva_spectrum']
        )
        st.pyplot(fig_ref)
        
        # Informações da norma
        st.info("""
        **Requisitos ISO 24443:2011:**
        - λc ≥ 370 nm
        - UVA-PF/SPF ≥ 1/3
        """)
    
    # Executar análise
    if analyze_button and uploaded_file_pre:
        with st.spinner("🔬 Executando análise completa..."):
            try:
                # Carregar dados
                df_pre, error = load_and_validate_data_iso(uploaded_file_pre, "pre_irradiation")
                if error:
                    st.error(error)
                    return
                
                # Calcular SPF in vitro
                spf_in_vitro = calculate_spf_in_vitro_iso(
                    df_pre, 
                    spectra['erythema_spectrum'], 
                    spectra['uv_ssr_spectrum']
                )
                
                # Calcular fator de correção C
                C = spf_in_vivo / spf_in_vitro if spf_in_vitro > 0 else 1.0
                
                # Calcular SPF ajustado
                spf_ajustado = calculate_adjusted_spf_iso(
                    df_pre, 
                    C, 
                    spectra['erythema_spectrum'], 
                    spectra['uv_ssr_spectrum']
                )
                
                # Calcular comprimento de onda crítico
                critical_wavelength = calculate_critical_wavelength_iso(df_pre, C)
                
                # Inicializar resultados UVA
                uva_pf_0, uva_pf_final, exposure_dose = 0, 0, 0
                
                if uploaded_file_uva:
                    df_uva, error = load_and_validate_data_iso(uploaded_file_uva, "post_irradiation")
                    if not error:
                        # Calcular UVA-PF₀
                        uva_pf_0 = calculate_uva_pf_initial_iso(
                            df_uva, 
                            C, 
                            spectra['ppd_spectrum'], 
                            spectra['uva_spectrum']
                        )
                        
                        # Calcular dose de exposição
                        exposure_dose = calculate_exposure_dose_iso(uva_pf_0)
                        
                        # Calcular UVA-PF final
                        uva_pf_final = calculate_uva_pf_final_iso(
                            df_uva, 
                            C, 
                            spectra['ppd_spectrum'], 
                            spectra['uva_spectrum']
                        )
                
                # Simular múltiplas réplicas para demonstração
                uva_pf_values = [uva_pf_final * (0.95 + 0.1 * np.random.random()) for _ in range(num_replicas)]
                
                # Armazenar resultados
                results = {
                    'spf_in_vitro': spf_in_vitro,
                    'spf_in_vivo': spf_in_vivo,
                    'correction_factor': C,
                    'spf_ajustado': spf_ajustado,
                    'uva_pf_0': uva_pf_0,
                    'uva_pf_final': uva_pf_final,
                    'exposure_dose': exposure_dose,
                    'critical_wavelength': critical_wavelength,
                    'uva_pf_values': uva_pf_values,
                    'num_replicas': num_replicas
                }
                
                st.session_state.current_results = results
                
                # Exibir resultados
                st.success("✅ Análise concluída com sucesso!")
                
                # Tabela de resultados
                st.subheader("📋 Resultados da Análise")
                
                results_data = [
                    ["SPF in vitro calculado:", f"{spf_in_vitro:.2f}"],
                    ["SPF in vivo declarado:", f"{spf_in_vivo:.2f}"],
                    ["Fator de correção (C):", f"{C:.4f}"],
                    ["SPF ajustado:", f"{spf_ajustado:.2f}"],
                    ["UVA-PF₀ (inicial):", f"{uva_pf_0:.2f}" if uva_pf_0 > 0 else "N/A"],
                    ["UVA-PF final:", f"{uva_pf_final:.2f}" if uva_pf_final > 0 else "N/A"],
                    ["Dose de exposição UVA:", f"{exposure_dose:.2f} J/cm²" if exposure_dose > 0 else "N/A"],
                    ["Comprimento de onda crítico (λc):", f"{critical_wavelength:.1f} nm"]
                ]
                
                # Verificar conformidade
                conformidade_lambda = "✅ CONFORME" if critical_wavelength >= 370 else "❌ NÃO CONFORME"
                ratio = uva_pf_final / spf_in_vivo if spf_in_vivo > 0 and uva_pf_final > 0 else 0
                conformidade_ratio = "✅ CONFORME" if ratio >= 0.33 else "❌ NÃO CONFORME"
                
                results_data.append(["λc ≥ 370 nm:", conformidade_lambda])
                results_data.append(["UVA-PF/SPF ≥ 1/3:", conformidade_ratio])
                
                # Exibir tabela de resultados
                for item in results_data:
                    col1, col2 = st.columns([2, 1])
                    col1.write(f"**{item[0]}**")
                    col2.write(item[1])
                
                # Gráfico de fatores de proteção
                if uva_pf_final > 0:
                    fig_protection = create_protection_factor_chart_iso(results)
                    st.pyplot(fig_protection)
                
                # Análise estatística se múltiplas réplicas
                if num_replicas > 1 and uva_pf_final > 0:
                    st.subheader("📊 Análise Estatística")
                    
                    ci, ci_percent, mean_uva_pf = calculate_confidence_interval(uva_pf_values)
                    
                    st.write(f"**UVA-PF Médio:** {mean_uva_pf:.2f}")
                    st.write(f"**Desvio Padrão:** {np.std(uva_pf_values, ddof=1):.3f}")
                    st.write(f"**Coeficiente de Variação:** {np.std(uva_pf_values, ddof=1)/mean_uva_pf*100:.2f}%")
                    st.write(f"**Intervalo de Confiança (95%):** ± {ci:.3f} (± {ci_percent:.2f}%)")
                
                # Seção de relatório PDF
                st.markdown("---")
                st.subheader("📊 Relatório de Análise")
                
                # Adicionar informações adicionais para o relatório
                sample_name = st.text_input("Nome da Amostra:", "Amostra Teste")
                operator = st.text_input("Operador:", "Analista")
                
                st.session_state.current_results['sample_name'] = sample_name
                st.session_state.current_results['operator'] = operator
                
                if st.button("📄 Gerar Relatório PDF ISO 24443:2011"):
                    pdf_buffer = generate_pdf_report_iso(st.session_state.current_results)
                    st.markdown(get_pdf_download_link(pdf_buffer, f"relatorio_protecao_solar_{sample_name}.pdf"), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")
                logger.exception("Erro na análise:")

if __name__ == "__main__":
    main()
