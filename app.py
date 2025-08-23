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
    total_numerator, total_denominator =  0, 0
    
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

# FUNÇÕES DE CÁLCULO MANSUR (ADICIONADAS)
def calculate_spf_mansur_simplified(df):
    """Calcula SPF usando a fórmula simplificada de Mansur et al. (1986)"""
    mansur_table = {
        290: 0.0150, 295: 0.0817, 300: 0.2874, 305: 0.3278,
        310: 0.1864, 315: 0.0839, 320: 0.0180
    }
    
    total_sum = 0.0
    for wavelength, eei_value in mansur_table.items():
        # Encontrar o valor mais próximo no DataFrame
        closest_row = df.iloc[(df['Comprimento de Onda'] - wavelength).abs().argsort()[:1]]
        if not closest_row.empty:
            absorbance = closest_row['A0i(λ)'].values[0]
            total_sum += eei_value * absorbance
    
    return 10 * total_sum

def calculate_spf_mansur_precise(df):
    """Calcula SPF Mansur com interpolação mais precisa"""
    mansur_table = {
        290: 0.0150, 295: 0.0817, 300: 0.2874, 305: 0.3278,
        310: 0.1864, 315: 0.0839, 320: 0.0180
    }
    
    total_sum = 0.0
    
    # Garantir que os dados estão ordenados por comprimento de onda
    df_sorted = df.sort_values('Comprimento de Onda')
    wavelengths_df = df_sorted['Comprimento de Onda'].values
    absorbance_df = df_sorted['A0i(λ)'].values
    
    # Verificar se há dados suficientes para interpolação
    if len(wavelengths_df) > 1 and len(np.unique(wavelengths_df)) > 1:
        try:
            # Criar função de interpolação
            interpolate_func = CubicSpline(wavelengths_df, absorbance_df)
            
            for wavelength, eei_value in mansur_table.items():
                if wavelength >= wavelengths_df.min() and wavelength <= wavelengths_df.max():
                    absorbance = interpolate_func(wavelength)
                    total_sum += eei_value * absorbance
        except:
            # Fallback para o método simplificado se a interpolação falhar
            st.warning("⚠️ Interpolação cúbica falhou. Usando método simplificado.")
            return calculate_spf_mansur_simplified(df)
    else:
        # Se não houver dados suficientes, usar método simplificado
        return calculate_spf_mansur_simplified(df)
    
    return 10 * total_sum

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
            required = ['Comprimento de Onda', 'P(λ)', 'I(极 λ)', 'Ai(λ)', 'A0i(λ)']
        
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
    ax.set_xlabel('Comprimento de Onda (nm)', fontsize极, 12, fontweight='bold')
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
    
    plt极.tight_layout()
    return fig

# GERAÇÃO DE RELATÓRIO PDF CONFORME ISO
def generate_pdf_report_iso(results):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Cabeçalho
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20*mm, height-20*mm, "RELATÓRIO DE ANÁLISE UVA-PF - ISO 24443:2011")
    c.setFont("Helvetica", 10)
    c.drawString(20*mm, height-25*mm, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    c.drawString(20*mm, height-30*mm, "Método in vitro para determinação da proteção UVA de produtos de proteção solar")
    
    # Resultados
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, height-45*mm, "RESULTADOS OBTIDOS:")
    c.setFont("Helvetica", 10)
    
    y = height - 50*mm
    for label, value in [
        ("UVA-PF₀ (Eq. 3)", f"{results.get('uva_pf_0', 0):.2f}"),
        ("UVA-PF Final (Eq. 5)", f"{results.get('uva_pf_final', 0):.2f}"),
        ("Dose de Exposição (Eq. 4)", f"{results.get('dose', 0):.2f} J/cm²"),
        ("λ Crítico", f"{results.get('critical_wavelength', 0):.1f} nm"),
        ("Coeficiente C", f"{results.get('C_value', 0):.极 4f}"),
        ("SPF in vitro (Eq. 1)", f"{results.get('spf_in_vitro', 0):.2f}"),
        ("SPF in vivo", f"{results.get('spf_in_vivo', 0):.2f}")
    ]:
        c.drawString(25*mm, y, f"{label}: {value}")
        y -= 5*mm
    
    # Informações de conformidade
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y-10*mm, "CONFORMIDADE COM ISO 24443:2011:")
    c.setFont("Helvetica", 10)
    
    # Verificar λ crítico
    critical_wl = results.get('critical_wavelength', 0)
    if critical_wl >= 370:
        c.drawString(25*mm, y-15* mm, "✅ λ Crítico ≥ 370 nm (Conforme requisito da norma)")
    else:
        c.drawString(25*mm, y-15*mm, "❌ λ Crítico < 370 nm (Não conforme)")
    
    # Verificar faixa UVA-PF para referência S2 (se aplicável)
    uva_pf_final = results.get('uva_pf_final', 0)
    if 10.7 <= uva_pf_final <= 14.7:
        c.drawString(25*mm, y-20*mm, "✅ UVA-PF dentro da faixa de referência S2 (10.7-14.7)")
    else:
        c.drawString(25*mm, y-20*mm, "⚠️ UVA-PF fora da faixa de referência S2")
    
    c.save()
    buffer.seek(0)
    return buffer

# INTERFACE PRINCIPAL CONFORME ISO 24443:2011
def main():
    st.title("🌞 Análise de Proteção Solar - ISO 24443:2011")
    
    # Carregar espectros de referência
    spectra = get_reference_spectra()
    wavelengths = spectra['wavelengths']
    ppd_spectrum = spectra['ppd_spectrum']
    erythema_spectrum = spectra['erythema_spectrum']
    uv_ssr_spectrum = spectra['uv_ssr_spectrum']
    uva_spectrum = spectra['uva_spectrum']
    
    with st.sidebar:
        st.title("Navegação - ISO 24443:2011")
        page = st.radio("Selecione:", ["Análise Completa", "Validação de Dados", "Sobre a Norma", "Espectros de Referência"])
        
        st.markdown("---")
        st.info("""
        **📋 Requisitos ISO 24443:2011:**
        - **SPF:** Comprimento de Onda, A0i(λ) [290-400nm]
        - **UVA:** Comprimento de Onda, P(λ), I(λ), Ai(λ), A0i(λ) [320-400nm]
        - **Aplicação:** 1.3 mg/cm² em placa PMMA
        """)
    
    if page == "Análise Completa":
        st.header("Análise Completa - ISO 24443:2011")
        
        tab1, tab2, tab3 = st.tabs(["📊 SPF Inicial", "🔬 UVA", "📈 Resultados"])
        
        with tab1:
            st.subheader("Cálculo do SPF in vitro (Eq. 1-2) + Mansur")
            
            with st.expander("ℹ️ Instruções Conforme ISO 24443:6.1-6.7"):
                st.markdown("""
                **Procedimento para análise SPF:**
                1. Faça upload de dados de absorbância inicial (A0i(λ))
                2. Dados devem cobrir 290-400nm em incrementos de 1nm
                3. Densidade de aplicação: 1.3 mg/cm²
                4. Placa PMMA com superfície rugosa
                5. Temperatura: 25-35°C durante secagem e irradiação
                """)
            
            uploaded_file_spf = st.file_uploader("Dados PRÉ-IRRADIAÇÃO (A0i(λ))", type=["csv", "xlsx"], key="spf_upload")
            
            if uploaded_file_spf:
                df_spf, error = load_and_validate_data_iso(uploaded_file_spf, "pre_irradiation")
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Dados SPF validados conforme ISO 24443!")
                    
                    # Visualização dos dados
                    with st.expander("📋 Visualizar dados carregados"):
                        st.dataframe(df_spf.head(10))
                        st.write(f"**Estatísticas:** {len(df_spf)} pontos, {df_spf['Comprimento de Onda'].min():.0f}-{df_spf['Comprimento de Onda'].max():.0f}nm")
                    
                    # Cálculo SPF in vitro
                    spf_in_vitro = calculate_spf_in_vitro_iso(df_spf, erythema_spectrum, uv_ssr_spectrum)
                    
                    # ⭐⭐ CÁLCULO DE MANSUR ADICIONADO AQUI ⭐⭐
                    spf_mansur = calculate_spf_mansur_simplified(df_spf)
                    spf_mansur_precise = calculate_spf_mansur_precise(df_spf)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("SPF in vitro (Eq. 1)", f"{spf_in_vitro:.2f}",
                                 help="Fator de proteção solar calculado in vitro")
                    
                    with col2:
                        st.metric("SPF Mansur Simplificado", f"{spf_mansur:.2f}",
                                 help="Método Mansur et al. (1986) - aproximação")
                    
                    with col3:
                        st.metric("SPF Mansur Interpolado", f"{spf_mansur_precise:.2f}",
                                 help="Método Mansur com interpolação cúbica")
                    
                    SPF_in_vivo = st.number_input("SPF in vivo medido:", 
                                                min_value=1.0, value=30.0, step=0.1,
                                                help="Valor do SPF determinado in vivo para calibrar o coeficiente C")
                    
                    # Cálculo do coeficiente C
                    def error_function(C):
                        return abs(calculate_adjusted_spf_iso(df_spf, C, erythema_spectrum, uv_ssr_spectrum) - SPF_in_vivo)
                    
                    result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                    C_value = result.x
                    spf_ajustado = calculate_adjusted_spf_iso(df_spf, C_value, erythema_spectrum, uv_ssr_spectrum)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Coeficiente C (Eq. 2)", f"{C_value:.4f}",
                                 help="Fator de ajuste para equalizar SPF in vitro/in vivo")
                    with col2:
                        st.metric("SPF ajustado (Eq. 2)", f"{spf_ajustado:.2f}")
                    
                    # Verificação do coeficiente C conforme ISO
                    if 0.8 <= C_value <= 1.6:
                        st.success("✅ Coeficiente C dentro da faixa válida (0.8-1.6)")
                    else:
                        st.warning("⚠️ Coeficiente C fora da faixa recomendada. Verifique a aplicação do produto.")
                    
                    # Gráfico de absorbância inicial
                    st.subheader("📊 Visualização dos Dados")
                    fig_absorbance = create_absorbance_plot_iso(df_spf)
                    st.pyplot(fig_absorbance)
                    
                    st.session_state.current_results.update({
                        'spf_in_vitro': spf_in_vitro,
                        'spf_mansur': spf_mansur,
                        'spf_mansur_precise': spf_mansur_precise,
                        'spf_in_vivo': SPF_in_vivo,
                        'C_value': C_value,
                        'spf_ajustado': spf_ajustado,
                        'dados_pre': df_spf
                    })
        
        with tab2:
            st.subheader("Análise UVA (Eq. 3-5)")
            
            if 'C_value' not in st.session_state.current_results:
                st.warning("⏳ Calcule primeiro o SPF para obter o coeficiente C")
            else:
                C_value = st.session_state.current_results['C_value']
                st.success(f"📊 Coeficiente C: {C_value:.4f}")
                
                with st.expander("ℹ️ Instruções Conforme ISO 24443:6.8-6.10"):
                    st.markdown("""
                    **Procedimento para análise UVA:**
                    1. Faça upload de dados completos de pós-irradiação
                    2. Dados devem conter: P(λ), I(λ), Ai(λ), A0i(λ)
                    极 3. Faixa espectral: 320-400nm
                    4. Dose de exposição: UVA-PF₀ × 1.2 J/cm²
                    5. Temperatura controlada: 25-35°C
                    """)
                
                uploaded_file_uva = st.file_uploader("Dados PÓS-IRRADIAÇÃO (P(λ), I(λ), Ai(λ), A0i(λ))", 
                                                    type=["csv", "xlsx"], key="uva_upload")
                
                if uploaded_file_uva:
                    df_uva, error = load_and_validate_data_iso(uploaded_file_uva, "post_irradiation")
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        is_valid, validation_msg = validate_uva_data_iso(df_uva)
                        if not is_valid:
                            st.error(f"❌ {validation_msg}")
                        else:
                            st.success("✅ Dados UVA validados conforme ISO 24443!")
                            
                            # Visualização dos dados
                            with st.expander("📋 Visualizar dados carregados"):
                                st.dataframe(df_uva.head(10))
                                st.write(f"极 **Estatísticas UVA:** {len(df_uva)} pontos, {df_uva['Comprimento de Onda'].min():.0极 f}-{df_uva['Comprimento de Onda'].max():.0f}nm")
                            
                            # Cálculos UVA conforme ISO
                            uva_pf_0 = calculate_uva_pf_initial_iso(df_uva, C_value, ppd_spectrum, uva_spectrum)
                            dose = calculate_exposure_dose_iso(uva_pf_0)
                            uva_pf_final = calculate_uva_pf_final_iso(df_uva, C_value, ppd_spectrum, uva_spectrum)
                            critical_wl = calculate_critical_wavelength_iso(df_uva, C_value)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: 
                                st.metric("UVA-PF₀ (Eq. 3)", f"{uva_pf_0:.2f}",
                                         help="Fator de Proteção UVA inicial")
                            with col2: 
                                st.metric("Dose (Eq. 4)", f"{dose:.2f} J/cm²",
                                         help="Dose de exposição UVA necessária")
                            with col3: 
                                st.metric("UVA-PF (Eq. 5)", f"{uva_pf_final:.2f}",
                                         help="Fator de Proteção UVA final")
                            with col4: 
                                status = "✅" if critical_wl >= 370 else "⚠️"
                                st.metric("λ Crítico", f"{critical_wl:.1f} nm", 
                                         f"{status} {'≥370nm' if critical_wl >= 370 else '<370nm'}")
                            
                            # Verificações de conformidade
                            if 10.7 <= uva_pf_final <= 14.7:
                                st.success("✅ UVA-PF dentro da faixa do padrão de referência S2 (10.7-14.7)")
                            else:
                                st.warning("⚠️ UVA-PF fora da faixa do padrão S2")
                            
                            if critical_w l >= 370:
                                st.success("✅ λ Crítico ≥ 370 nm (Conforme requisito ISO)")
                            else:
                                st.error("❌ λ Crítico < 370 nm (Não conforme)")
                            
                            # Gráficos para análise UVA
                            st.subheader("📊 Visualização dos Dados UVA")
                            
                            fig_absorbance_uva = create_absorbance_plot_iso(
                                st.session_state.current_results['dados_pre'], 
                                df_uva, 
                                critical_wl
                            )
                            st.pyplot(fig_absorbance_uva)
                            
                            st.session_state.current_results.update({
                                'uva_pf_0': uva_pf_0,
                                'dose': dose,
                                'uva_pf_final': uva_pf_final,
                                'critical_wavelength': critical_wl,
                                'dados_post': df_uva
                            })
        
        with tab3:
            st.subheader("📈 Resultados Completos")
            
            if 'uva_pf_final' not in st.session_state.current_results:
                st.warning("⏳ Complete as análises anteriores para ver os resultados completos")
            else:
                results = st.session_state.current_results
                
                st.success("✅ Análise concluída com sucesso!")
                
                # Resumo dos resultados
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("📊 Resultados SPF")
                    st.metric("SPF in vitro (Eq. 1)", f"{results['spf_in_vitro']:.2f}")
                    st.metric("SPF Mansur Simplificado", f"{results['spf_mansur']:.2f}")
                    st.metric("SPF Mansur Interpolado", f"{results['spf_mansur_precise']:.2f}")
                
                with col2:
                    st.subheader("🎯 Resultados Ajustados")
                    st.metric("SPF in vivo", f"{results['spf_in_vivo']:.2f}")
                    st.metric("SPF ajustado (Eq. 2)", f"{results['spf_ajustado']:.2f}")
                    st.metric("Coeficiente C", f"{results['C_value']:.4f}")
                
                with col3:
                    st.subheader("🌅 Resultados UVA")
                    st.metric("UVA-PF₀ (Eq. 3)", f"{results['uva_pf_0']:.2f}")
                    st.metric("UVA-PF Final (Eq. 5)", f"{results['uva_pf_final']:.极 2f}")
                    st.metric("Dose de Exposição (Eq. 4)", f"{results['dose']:.2f} J/cm²")
                    st.metric("λ Crítico", f"{results['critical_wavelength']:.1f} nm")
                
                # Gráfico comparativo
                st.subheader("📊 Comparação dos Fatores de Proteção")
                fig_comparison = create_protection_factor_chart_iso(results)
                st.pyplot(fig_comparison)
                
                # Interpretação dos resultados
                st.subheader("🔍 Interpretação dos Resultados")
                
                # Verificação do λ crítico
                if results['critical_wavelength'] >= 370:
                    st.success("✅ **λ Crítico**: Atende ao requisito mínimo de 370nm (ISO 24443:2011)")
                else:
                    st.error("❌ **λ Crítico**: Não atende ao requisito mínimo de 370nm")
                
                # Verificação da relação UVA-PF/SPF
                uva_pf_ratio = results['uva_pf_final'] / results['spf_in_vivo'] if results['spf_in_vivo'] > 0 else 0
                if uva_pf_ratio >= 0.33:
                    st.success(f"✅ **Relação UVA-PF/SPF**: {uva_pf_ratio:.3f} (≥ 1/3, atendendo ao requisito da UE)")
                else:
                    st.warning(f"⚠️ **Relação UVA-PF/SPF**: {uva_pf_ratio:.3f} (< 1/3, não atendendo ao requisito da UE)")
                
                # Relatório PDF
                st.subheader("📄 Relatório de Análise")
                pdf_buffer = generate_pdf_report_iso(results)
                st.download_button(
                    label="📥 Baixar Relatório PDF (ISO 24443:2011)",
                    data=pdf_buffer,
                    file_name=f"relatorio_uva_pf_iso24443_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                
                # Explicação das Fórmulas
                with st.expander("🧮 Detalhes das Fórmulas Utilizadas - ISO 24443:2011"):
                    st.markdown("""
                    ### **Fórmulas ISO 24443:2011**
                    
                    **Eq. 1 - SPF in vitro**: 
                    `SPF = ∫ E(λ)·I(λ) dλ / ∫ E(λ)·I(λ)·10^(-A0i(λ)) dλ`
                    
                    **Eq. 2 - SPF ajustado**: 
                    `SPF_ajustado = ∫ E(λ)·I(λ) dλ / ∫ E(λ)·I(λ)·10^(-A0i(λ)·C) dλ`
                    
                    **Eq. 3 - UVA-PF₀**: 
                    `UVA-PF₀ = ∫ P(λ)·I(λ) dλ / ∫ P(λ)·I(λ)·10^(-A0i(λ)·C) dλ`
                    
                    **Eq. 4 - Dose**: 
                    `Dose = UVA-PF₀ × 1.2 J/cm²`
                    
                    **Eq. 5 - UVA-PF final**: 
                    `UVA-PF = ∫ P(λ)·I(λ) dλ / ∫ P(λ)·I(λ)·10^(-Ai(λ)·C) dλ`
                    
                    **Onde:**
                    - `E(λ)`: Espectro de ação para eritema (CIE 1987)
                    - `P(λ)`: Espectro de ação para PPD
                    - `I(λ)`: Espectro de irradiância
                    - `A0i(λ)`: Absorbância inicial
                    - `Ai(λ)`: Absorbância após irradiação
                    - `C`: Coeficiente de ajuste
                    """)
    
    elif page == "Validação de Dados":
        st.header("✅ Validação de Dados - ISO 24443:2011")
        
        st.info("""
        Esta seção permite validar seus dados antes da análise completa conforme ISO 24443:2011.
        Verifique se os arquivos atendem aos requisitos da norma.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Validação de Dados SPF")
            spf_val_file = st.file_uploader("Dados para validação SPF", type=["csv", "xlsx"], key="val_spf")
            if spf_val_file:
                df_val, error = load_and_validate_data_iso(spf_val_file, "pre_irradiation")
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Dados SPF válidos conforme ISO 24443!")
                    st.write(f"**Formato:** {len(df_val)} pontos espectrais")
                    st.write(f"**Faixa espectral:** {df_val['Comprimento de Onda'].min():.0f}-{df_val['Comprimento de Onda'].max():.0f} nm")
                    
                    # Verificação de faixa espectral
                    if df_val['Comprimento de Onda'].min() <= 290 and df_val['极 Comprimento de Onda'].max() >= 400:
                        st.success("✅ Faixa espectral completa (290-400nm)")
                    else:
                        st.warning("⚠️ Faixa espectral incompleta")
                    
                    # Visualização rápida
                    fig_val = create_absorbance_plot_iso(df_val)
                    st.pyplot(fig_val)
        
        with col2:
            st.subheader("🌅 Validação de Dados UVA")
            uva_val_file = st.file_uploader("Dados para validação UVA", type=["csv", "xlsx"], key="val_uva")
            if uva_val_file:
                df_val, error = load_and_validate_data_iso(uva_val_file, "post_irradiation")
                if error:
                    st.error(f"❌ {error}")
                else:
                    is_valid, validation_msg = validate_uva_data_iso(df_val)
                    if not is_valid:
                        st.error(f"❌ {validation_msg}")
                    else:
                        st.success("✅ Dados UVA válidos conforme ISO 24443!")
                        st.write(f"**Formato:** {len(df_val)} pontos espectrais")
                        st.write(f"**Faixa UVA:** {df_val['Comprimento de Onda'].min():.0f}-{df_val['Comprimento de Onda'].max():.0f} nm")
                        
                        # Verificar colunas presentes
                        present_cols = [col for col in ['P(λ)', 'I(λ)', 'Ai(λ)', 'A0i(λ)'] if col in df_val.columns]
                        st.write(f"**Colunas detectadas:** {', '.join(present_cols)}")
                        
                        # Verificação de faixa UVA
                        if df_val['Comprimento de Onda'].min() <= 320 and df_val['Comprimento de Onda'].max() >= 400:
                            st.success("✅ Faixa UVA completa (320-400nm)")
                        else:
                            st.warning("⚠️ Faixa UVA incompleta")
    
    elif page == "Espectros de Referência":
        st.header("📊 Espectros de Referência - ISO 24443:2011 Anexo C")
        
        st.info("""
        Espectros de referência utilizados nos cálculos conforme Anexo C da ISO 24443:2011.
        Estes valores são normalizados e utilizados nas equações da norma.
        """)
        
        # Mostrar espectros de referência
        fig_ref = create_reference_spectra_plot_iso(wavelengths, ppd_spectrum, erythema_spectrum, uva_spectrum)
        st.pyplot(fig_ref)
        
        # Tabela com valores de referência
        with st.expander("📋 Valores Numéricos dos Espectros de Referência"):
            ref_df = pd.DataFrame({
                'Comprimento de Onda (nm)': wavelengths,
                'PPD (P(λ))': ppd_spectrum,
                'Eritema CIE (E(λ))': erythema_spectrum,
                'UV-SSR (I(λ)) W/m²nm': uv_ssr_spectrum,
                'UVA (I(λ)) W/m²nm': uva_spectrum
            })
            st.dataframe(ref_df.head(20))
            st.write(f"**Total:** {len(ref_df)} pontos de 290-400nm")
    
    else:
        st.header("📋 Sobre a Norma ISO 24443:2011")
        
        st.markdown("""
        ## 📋 Informações sobre a Norma ISO 24443:2011
        
        **Cosméticos — Método de ensaio de proteção solar — Determinação in vitro da fotoproteção UVA**
        
        ### 🔍 Escopo da Norma:
        
        Esta norma especifica um procedimento *in vitro* para caracterizar a proteção UVA de produtos de proteção solar. 
        O método fornece uma curva de absorbância espectral UV a partir da qual vários parâmetros de proteção UVA podem ser calculados.
        
        ### 📊 Principais Parâmetros Avaliados:
        
        - **UVA-PF**: Fator de Proteção UVA (correlaciona com *in vivo* PPD)
        - **λ Crítico**: Comprimento de onda abaixo do qual 90% da absorbância total é obtida
        - **Proporcionalidade de absorbância UVA**
        
        ### ⚠️ Limitações:
        
        - Não aplicável a produtos em pó
        - Baseia-se in resultados SPF *in vivo* para escalonamento
        - Não é um método totalmente *in vitro*
        
        ### 🧪 Requisitos do Método:
        
        - **Placa substrato**: PMMA com superfície rugosa
        - **Densidade de aplicação**: 1.3 mg/cm²
        - **Faixa espectral**: 290-400 nm
        - **Temperatura**: 25-35°C
        - **Dose de exposição**: UVA-PF₀ × 1.2 J/cm²
        
        ### ✅ Critérios de Aceitação:
        
        - **极 λ Crítico** deve ser ≥ 370 nm
        - **Referência S2**: UVA-PF entre 10.7-14.7
        - **Coeficiente C**: 0.8-1.6
        
        ### 📈 Espectros de Referência (Anexo C):
        
        - **E(λ)**: Espectro de ação para eritema (CIE 1987)
        - **P(λ)**: Espectro de ação para PPD (Persistent Pigment Darkening)
        - **I(λ)**: Espectro de irradiância solar
        
        ### 🔬 Equipamentos Requeridos:
        
        - Espectrofotômetro UV (290-400 nm, incremento de 1 nm)
        - Fonte de exposição UV com espectro definido
        - Placas PMMA qualificadas
        - Sistema de controle de temperatura
        """)
        
        # Referências
        with st.expander("📚 Referências Normativas"):
            st.markdown("""
            - **ISO 17025**: Requisitos gerais para competência de laboratórios
            - **CIE S007/ISO 17166**: Espectro de ação para eritema
            - **COLIPA (1994)**: Método de ensaio SPF
            - **DIN 67501**: Avaliação de produtos de proteção solar
            """)

if __name__ == "__main__":
    main()
