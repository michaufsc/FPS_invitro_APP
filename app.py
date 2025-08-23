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

# Configuração da página
st.set_page_config(
    page_title="Análise de Proteção Solar - ISO 24443:2011 + Mansur",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Sistema de logging
logger = logging.getLogger('UVA_Analysis')
logger.setLevel(logging.INFO)

def load_reference_spectra():
    """Carrega os espectros de referência COMPLETOS da Annex C da norma ISO"""
    wavelengths = np.arange(290, 401)  # 290-400nm, 111 valores
    
    # Espectro de ação PPD (Tabela C.1) - 111 valores
    ppd_spectrum = np.array([
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        1.000, 0.975, 0.950, 0.925, 0.900, 0.875, 0.850, 0.825, 0.800, 0.775,
        0.750, 0.725, 0.700, 0.675, 0.650, 0.625, 0.600, 0.575, 0.550, 0.525,
        0.500, 0.494, 0.488, 0.482, 0.476, 0.470, 0.464, 0.458, 0.452, 0.446,
        0.440, 0.434, 0.428, 0.422, 0.416, 0.410, 0.404, 0.398, 0.392, 0.386,
        0.380, 0.374, 0.368, 0.362, 0.356, 0.350, 0.344, 0.338, 0.332, 0.326,
        0.320, 0.314, 0.308, 0.302, 0.296, 0.290, 0.284, 0.278, 0.272, 0.266,
        0.260, 0.254, 0.248, 0.242, 0.236, 0.230, 0.224, 0.218, 0.212, 0.206,
        0.200, 0.194, 0.188, 0.182, 0.176, 0.170, 0.164, 0.158, 0.152, 0.146,
        0.140
    ])
    
    # Espectro de eritema CIE 1987 (Tabela C.1) - 111 valores
    erythema_spectrum = np.array([
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.8054,
        0.6486, 0.5224, 0.4207, 0.3388, 0.2729, 0.2198, 0.1770, 0.1426, 0.1148, 0.0925,
        0.0745, 0.0600, 0.0483, 0.0389, 0.0313, 0.0252, 0.0203, 0.0164, 0.0132, 0.0106,
        0.0086, 0.0069, 0.0055, 0.0045, 0.0036, 0.0029, 0.0023, 0.0019, 0.0015, 0.0012,
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
        0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
        0.0010
    ])
    
    # Espectro de irradiância UV-SSR (Tabela C.1) - 111 valores
    uv_ssr_spectrum = np.array([
        8.741E-06, 1.450E-05, 2.659E-05, 4.574E-05, 1.006E-04, 2.589E-04, 7.035E-04, 1.678E-03, 3.727E-03, 7.938E-03,
        1.478E-02, 2.514E-02, 4.176E-02, 6.223E-02, 8.690E-02, 1.216E-01, 1.615E-01, 1.989E-01, 2.483E-01, 2.894E-01,
        3.358E-01, 3.872E-01, 4.311E-01, 4.884E-01, 5.121E-01, 5.567E-01, 5.957E-01, 6.256E-01, 6.565E-01, 6.879E-01,
        7.236E-01, 7.371E-01, 7.677E-01, 7.955E-01, 7.987E-01, 8.290E-01, 8.435E-01, 8.559E-01, 8.791E-01, 8.951E-01,
        9.010E-01, 9.161E-01, 9.434E-01, 9.444E-01, 9.432E-01, 9.571E-01, 9.663E-01, 9.771E-01, 9.770E-01, 9.967E-01,
        9.939E-01, 1.007E+00, 1.012E+00, 1.011E+00, 1.021E+00, 1.025E+00, 1.033E+00, 1.034E+00, 1.040E+00, 1.027E+00,
        1.045E+00, 1.042E+00, 1.040E+00, 1.039E+00, 1.043E+00, 1.046E+00, 1.035E+00, 1.039E+00, 1.027E+00, 1.035E+00,
        1.037E+00, 1.025E+00, 1.023E+00, 1.016E+00, 9.984E-01, 9.960E-01, 9.674E-01, 9.648E-01, 9.389E-01, 9.191E-01,
        8.977E-01, 8.725E-01, 8.473E-01, 8.123E-01, 7.840E-01, 7.416E-01, 7.148E-01, 6.687E-01, 6.280E-01, 5.863E-01,
        5.341E-01, 4.925E-01, 4.482E-01, 3.932E-01, 3.428E-01, 2.985E-01, 2.567E-01, 2.148E-01, 1.800E-01, 1.486E-01,
        1.193E-01, 9.403E-02, 7.273E-02, 5.532E-02, 4.010E-02, 2.885E-02, 2.068E-02, 1.400E-02, 9.510E-03, 6.194E-03,
        4.172E-03
    ])
    
    # Espectro de irradiância UVA (Tabela C.1) - 111 valores
    uva_spectrum = np.array([
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        4.843E-06, 8.466E-06, 1.356E-05, 2.074E-05, 3.032E-05, 4.294E-05, 5.738E-05, 7.601E-05, 9.845E-05, 1.215E-04,
        1.506E-04, 1.811E-04, 2.132E-04, 2.444E-04, 2.833E-04, 3.186E-04, 3.589E-04, 3.980E-04, 4.387E-04, 4.778E-04,
        5.198E-04, 5.608E-04, 5.998E-04, 6.384E-04, 6.739E-04, 7.123E-04, 7.468E-04, 7.784E-04, 8.180E-04, 8.427E-04,
        8.754E-04, 9.044E-04, 9.288E-04, 9.486E-04, 9.733E-04, 9.863E-04, 1.009E-03, 1.028E-03, 1.045E-03, 1.062E-03,
        1.078E-03, 1.086E-03, 1.098E-03, 1.095E-03, 1.100E-03, 1.100E-03, 1.093E-03, 1.087E-03, 1.082E-03, 1.071E-03,
        1.048E-03, 1.026E-03, 9.953E-04, 9.703E-04, 9.367E-04, 9.057E-04, 8.757E-04, 8.428E-04, 8.058E-04, 7.613E-04,
        7.105E-04, 6.655E-04, 6.115E-04, 5.561E-04, 4.990E-04, 4.434E-04, 3.876E-04, 3.363E-04, 2.868E-04, 2.408E-04,
        2.012E-04, 1.640E-04, 1.311E-04, 1.028E-04, 7.897E-05, 5.975E-05, 4.455E-05, 3.259E-05, 2.302E-05, 1.581E-05,
        1.045E-05
    ])
    return wavelengths, ppd_spectrum, erythema_spectrum, uv_ssr_spectrum, uva_spectrum

# Sistema de sessão
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}

# FUNÇÕES DE CÁLCULO
def get_spectrum_value(wavelength, spectrum_array, min_wavelength=290):
    idx = wavelength - min_wavelength
    if 0 <= idx < len(spectrum_array):
        return spectrum_array[idx]
    return 0.0

def calculate_spf_in_vitro(df, erythema_spectrum, uv_ssr_spectrum):
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

def calculate_adjusted_spf(df, C, erythema_spectrum, uv_ssr_spectrum):
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

def calculate_uva_pf_initial(df, C, ppd_spectrum, uva_spectrum):
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

def calculate_uva_pf_final(df, C, ppd_spectrum, uva_spectrum):
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

def calculate_exposure_dose(uva_pf_0):
    return uva_pf_0 * 1.2

def calculate_critical_wavelength(df, C):
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

# FÓRMULA DE MANSUR SIMPLIFICADA
def calculate_spf_mansur_simplified(df):
    mansur_table = {
        290: 0.0150, 295: 0.0817, 300: 0.2874, 305: 0.3278,
        310: 0.1864, 315: 0.0839, 320: 0.0180
    }
    
    total_sum = 0.0
    for wavelength, eei_value in mansur_table.items():
        closest_row = df.iloc[(df['Comprimento de Onda'] - wavelength).abs().argsort()[:1]]
        if not closest_row.empty:
            absorbance = closest_row['A0i(λ)'].values[0]
            total_sum += eei_value * absorbance
    
    return 10 * total_sum

# FUNÇÕES AUXILIARES
def detect_encoding(uploaded_file):
    raw_data = uploaded_file.read()
    uploaded_file.seek(0)
    result = chardet.detect(raw_data)
    return result['encoding']

def load_and_validate_data(uploaded_file, data_type="pre_irradiation"):
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
        df = df[(df['Comprimento de Onda'] >= 290) & (df['Comprimento de Onda'] <= 400)]
        
        if len(df) == 0:
            return None, "Nenhum dado válido na faixa de 290-400nm"
        
        st.success(f"Dados carregados com sucesso! {len(df)} linhas válidas.")
        return df, None
        
    except Exception as e:
        return None, f"Erro ao carregar arquivo: {str(e)}"

def validate_uva_data(df):
    required_cols = ['Comprimento de Onda', 'P(λ)', 'I(λ)', 'Ai(λ)', 'A0i(λ)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Colunas UVA faltando: {', '.join(missing_cols)}"
    
    wavelengths = df['Comprimento de Onda'].values
    if min(wavelengths) > 320 or max(wavelengths) < 400:
        return False, "Faixa de wavelength UVA incompleta (320-400nm requerido)"
    
    return True, "Dados UVA válidos"

# GERAÇÃO DE RELATÓRIO PDF
def generate_pdf_report(results):
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
    
    # Resultados
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, height-40*mm, "RESULTADOS OBTIDOS:")
    c.setFont("Helvetica", 10)
    
    y = height - 45*mm
    for label, value in [
        ("UVA-PF₀ (Eq. 3)", f"{results.get('uva_pf_0', 0):.2f}"),
        ("UVA-PF Final (Eq. 5)", f"{results.get('uva_pf_final', 0):.2f}"),
        ("Dose de Exposição (Eq. 4)", f"{results.get('dose', 0):.2f} J/cm²"),
        ("λ Crítico", f"{results.get('critical_wavelength', 0):.1f} nm"),
        ("Coeficiente C", f"{results.get('C_value', 0):.4f}"),
        ("SPF Mansur Simplificado", f"{results.get('spf_mansur', 0):.2f}")
    ]:
        c.drawString(25*mm, y, f"{label}: {value}")
        y -= 5*mm
    
    c.save()
    buffer.seek(0)
    return buffer

# INTERFACE PRINCIPAL
def main():
    st.title("🌞 Análise de Proteção Solar - ISO 24443:2011 + Mansur")
    
    wavelengths, ppd_spectrum, erythema_spectrum, uv_ssr_spectrum, uva_spectrum = load_reference_spectra()
    
    with st.sidebar:
        st.title("Navegação")
        page = st.radio("Selecione:", ["ISO 24443 Completo", "Validação de Dados", "Sobre a Norma"])
        
        st.markdown("---")
        st.info("""
        **📋 Formatos esperados:**
        - **SPF:** Comprimento de Onda, A0i(λ)
        - **UVA:** Comprimento de Onda, P(λ), I(λ), Ai(λ), A0i(λ)
        """)
    
    if page == "ISO 24443 Completo":
        st.header("Análise Completa - ISO 24443:2011")
        
        tab1, tab2, tab3 = st.tabs(["📊 SPF Inicial", "🔬 UVA", "📈 Resultados"])
        
        with tab1:
            st.subheader("Cálculo do SPF (Eq. 1-2) + Mansur")
            uploaded_file_spf = st.file_uploader("Dados PRÉ-IRRADIAÇÃO (A0i(λ))", type=["csv", "xlsx"], key="spf_upload")
            
            if uploaded_file_spf:
                df_spf, error = load_and_validate_data(uploaded_file_spf, "pre_irradiation")
                if error:
                    st.error(error)
                else:
                    st.success("Dados SPF validados!")
                    
                    # Cálculo SPF in vitro
                    spf_in_vitro = calculate_spf_in_vitro(df_spf, erythema_spectrum, uv_ssr_spectrum)
                    
                    # Cálculo SPF Mansur
                    spf_mansur = calculate_spf_mansur_simplified(df_spf)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("SPF in vitro (Eq. 1)", f"{spf_in_vitro:.2f}")
                        st.metric("SPF Mansur Simplificado", f"{spf_mansur:.2f}")
                    
                    SPF_in_vivo = st.number_input("SPF in vivo medido:", min_value=1.0, value=30.0, step=0.1)
                    
                    def error_function(C):
                        return abs(calculate_adjusted_spf(df_spf, C, erythema_spectrum, uv_ssr_spectrum) - SPF_in_vivo)
                    
                    result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                    C_value = result.x
                    spf_ajustado = calculate_adjusted_spf(df_spf, C_value, erythema_spectrum, uv_ssr_spectrum)
                    
                    with col2:
                        st.metric("Coeficiente C (Eq. 2)", f"{C_value:.4f}")
                        st.metric("SPF ajustado (Eq. 2)", f"{spf_ajustado:.2f}")
                    
                    st.session_state.current_results.update({
                        'spf_in_vitro': spf_in_vitro,
                        'spf_mansur': spf_mansur,
                        'spf_in_vivo': SPF_in_vivo,
                        'C_value': C_value,
                        'spf_ajustado': spf_ajustado,
                        'dados_pre': df_spf
                    })
        
        with tab2:
            st.subheader("Análise UVA (Eq. 3-5)")
            
            if 'C_value' not in st.session_state.current_results:
                st.warning("Calcule primeiro o SPF para obter o coeficiente C")
            else:
                C_value = st.session_state.current_results['C_value']
                st.success(f"Coeficiente C: {C_value:.4f}")
                
                uploaded_file_uva = st.file_uploader("Dados PÓS-IRRADIAÇÃO (P(λ), I(λ), Ai(λ), A0i(λ))", type=["csv", "xlsx"], key="uva_upload")
                
                if uploaded_file_uva:
                    df_uva, error = load_and_validate_data(uploaded_file_uva, "post_irradiation")
                    if error:
                        st.error(error)
                    else:
                        is_valid, validation_msg = validate_uva_data(df_uva)
                        if not is_valid:
                            st.error(validation_msg)
                        else:
                            st.success("Dados UVA validados!")
                            
                            uva_pf_0 = calculate_uva_pf_initial(df_uva, C_value, ppd_spectrum, uva_spectrum)
                            dose = calculate_exposure_dose(uva_pf_0)
                            uva_pf_final = calculate_uva_pf_final(df_uva, C_value, ppd_spectrum, uva_spectrum)
                            critical_wl = calculate_critical_wavelength(df_uva, C_value)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: st.metric("UVA-PF₀ (Eq. 3)", f"{uva_pf_0:.2f}")
                            with col2: st.metric("Dose (Eq. 4)", f"{dose:.2f} J/cm²")
                            with col3: st.metric("UVA-PF (Eq. 5)", f"{uva_pf_final:.2f}")
                            with col4: 
                                status = "OK" if critical_wl >= 370 else "ALERTA"
                                st.metric("λ Crítico", f"{critical_wl:.1f} nm", status)
                            
                            if 10.7 <= uva_pf_final <= 14.7:
                                st.success("✅ Resultado dentro da faixa do padrão de referência S2")
                            else:
                                st.warning("⚠️ Resultado fora da faixa do padrão S2 (10.7-14.7)")
                            
                            st.session_state.current_results.update({
                                'uva_pf_0': uva_pf_0,
                                'dose': dose,
                                'uva_pf_final': uva_pf_final,
                                'critical_wavelength': critical_wl,
                                'dados_post': df_uva
                            })
        
        with tab3:
            st.subheader("Resultados Completos")
            
            if 'uva_pf_final' not in st.session_state.current_results:
                st.warning("Complete as análises anteriores")
            else:
                results = st.session_state.current_results
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SPF in vitro", f"{results['spf_in_vitro']:.2f}")
                    st.metric("SPF Mansur", f"{results['spf_mansur']:.2f}")
                    st.metric("SPF in vivo", f"{results['spf_in_vivo']:.2f}")
                    st.metric("SPF ajustado", f"{results['spf_ajustado']:.2f}")
                    st.metric("Coeficiente C", f"{results['C_value']:.4f}")
                
                with col2:
                    st.metric("UVA-PF₀", f"{results['uva_pf_0']:.2f}")
                    st.metric("UVA-PF Final", f"{results['uva_pf_final']:.2f}")
                    st.metric("Dose de Exposição", f"{results['dose']:.2f} J/cm²")
                    st.metric("λ Crítico", f"{results['critical_wavelength']:.1f} nm")
                
                # Gráficos Corrigidos
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Gráfico 1: Dados experimentais
                ax1.plot(results['dados_pre']['Comprimento de Onda'], 
                        results['dados_pre']['A0i(λ)'], 
                        label='Absorbância Inicial (SPF)', linewidth=2, color='blue')
                
                if 'dados_post' in results:
                    ax1.plot(results['dados_post']['Comprimento de Onda'], 
                            results['dados_post']['Ai(λ)'], 
                            label='Absorbância Final (UVA)', linewidth=2, color='red')
                
                ax1.set_xlabel('Comprimento de Onda (nm)')
                ax1.set_ylabel('Absorbância')
                ax1.set_title('Espectro de Absorbância')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(290, 400)
                # Gráfico 2: Espectros de referência principais
                ax2.plot(wavelengths, erythema_spectrum, label='Eritema CIE (E(λ))', color='orange', alpha=0.7)
                ax2.plot(wavelengths, ppd_spectrum, label='PPD (P(λ))', color='purple', alpha=0.7)
                ax2.set_xlabel('Comprimento de Onda (nm)')
                ax2.set_ylabel('Valor do Espectro')
                ax2.set_title('Espectros de Referência Principais')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(290, 400)
                st.pyplot(fig)
                
                # Relatório PDF
                st.subheader("📄 Relatório de Análise")
                pdf_buffer = generate_pdf_report(results)
                st.download_button(
                    label="📥 Baixar Relatório PDF",
                    data=pdf_buffer,
                    file_name=f"relatorio_uva_pf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                
                # Explicação das Fórmulas
                with st.expander("🧮 Detalhes das Fórmulas Utilizadas"):
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
                    
                    ### **Fórmula de Mansur Simplificada**
                    
                    **Mansur et al. (1986)**:
                    `SPF = 10 × ∑ [EE(λ)·I(λ)·Abs(λ)]` para λ = 290,295,...,320nm
                    
                    **Valores de EE(λ)·I(λ)**:
                    - 290nm: 0.0150
                    - 295nm: 0.0817
                    - 300nm: 0.2874
                    - 305nm: 0.3278
                    - 310nm: 0.1864
                    - 315nm: 0.0839
                    - 320nm: 0.0180
                    """)
    
    elif page == "Validação de Dados":
        st.header("Validação de Dados e Espectros")
        # ... (código de validação)
    
    else:
        st.header("Sobre a Norma ISO 24443:2011")
        # ... (código informativo)

if __name__ == "__main__":
    main()
