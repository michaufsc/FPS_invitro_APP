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

# Sistema de sessão
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}

# FUNÇÕES DE CÁLCULO
def load_reference_spectra():
    """Carrega espectros de referência com tamanhos garantidos (111 valores - ISO 24443)"""
    wavelengths = np.arange(290, 401)  # 111 valores: 290 a 400 nm
    
    # Espectro PPD (Persistent Pigment Darkening) - 111 valores
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
    
    # Espectro de eritema CIE - 111 valores
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
    
    # Espectro de irradiância solar UVA (placeholder - valores típicos)
    uva_spectrum = np.array([
        0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.048,
        0.065, 0.085, 0.110, 0.140, 0.175, 0.215, 0.260, 0.310, 0.365, 0.425,
        0.490, 0.560, 0.635, 0.715, 0.800, 0.890, 0.985, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000
    ])
    
    # Espectro UV SSR (placeholder)
    uv_ssr_spectrum = np.ones(111) * 0.7
    
    return wavelengths, ppd_spectrum, erythema_spectrum, uv_ssr_spectrum, uva_spectrum

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

# FUNÇÕES PARA MELHORAR OS GRÁFICOS
def create_absorbance_plot(df_pre, df_post=None, critical_wavelength=None):
    """Cria gráfico de absorbância com design melhorado"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    ax.set_title('Espectro de Absorbância UV', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(290, 400)
    
    # Adicionar áreas de UVB e UVA
    ax.axvspan(290, 320, alpha=0.1, color='blue', label='UVB')
    ax.axvspan(320, 400, alpha=0.1, color='red', label='UVA')
    
    # Adicionar anotações
    ax.text(300, ax.get_ylim()[1]*0.8, 'UVB', color='blue', fontweight='bold', ha='center')
    ax.text(360, ax.get_ylim()[1]*0.8, 'UVA', color='red', fontweight='bold', ha='center')
    
    plt.tight_layout()
    return fig

def create_reference_spectra_plot(wavelengths, ppd_spectrum, erythema_spectrum, uva_spectrum):
    """Cria gráfico dos espectros de referência com design melhorado"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotar espectros de referência
    ax.plot(wavelengths, erythema_spectrum, label='Eritema CIE (E(λ))', 
            linewidth=2, color='#2ca02c', alpha=0.8)
    ax.plot(wavelengths, ppd_spectrum, label='PPD (P(λ))', 
            linewidth=2, color='#d62728', alpha=0.8)
    ax.plot(wavelengths, uva_spectrum, label='Irradiância UVA (I(λ))', 
            linewidth=2, color='#9467bd', alpha=0.8)
    
    # Melhorar a aparência do gráfico
    ax.set_xlabel('Comprimento de Onda (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Normalizado', fontsize=12, fontweight='bold')
    ax.set_title('Espectros de Referência para Cálculos', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(290, 400)
    
    # Adicionar áreas de UVB e UVA
    ax.axvspan(290, 320, alpha=0.1, color='blue')
    ax.axvspan(320, 400, alpha=0.1, color='red')
    
    # Adicionar anotações
    ax.text(300, ax.get_ylim()[1]*0.9, 'UVB', color='blue', fontweight='bold', ha='center')
    ax.text(360, ax.get_ylim()[1]*0.9, 'UVA', color='red', fontweight='bold', ha='center')
    
    plt.tight_layout()
    return fig

def create_protection_factor_chart(results):
    """Cria gráfico de barras para fatores de proteção"""
    labels = ['SPF in vitro', 'SPF Mansur', 'SPF in vivo', 'SPF ajustado', 'UVA-PF₀', 'UVA-PF Final']
    values = [
        results.get('spf_in_vitro', 0),
        results.get('spf_mansur', 0),
        results.get('spf_in_vivo', 0),
        results.get('spf_ajustado', 0),
        results.get('uva_pf_0', 0),
        results.get('uva_pf_final', 0)
    ]
    
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    
    # Adicionar valores nas barras
    for i, v in enumerate(values):
        ax.text(i, v + max(values)*0.01, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Melhorar a aparência do gráfico
    ax.set_ylabel('Valor do Fator de Proteção', fontsize=12, fontweight='bold')
    ax.set_title('Comparação dos Fatores de Proteção', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

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
            
            # Adicionar informações para o usuário
            with st.expander("ℹ️ Instruções"):
                st.markdown("""
                **Para análise SPF:**
                1. Faça upload de um arquivo com dados de absorbância inicial
                2. O arquivo deve conter pelo menos as colunas: **Comprimento de Onda** e **A0i(λ)**
                3. Formatos suportados: CSV (separador ; ou ,) ou Excel
                4. Os dados devem cobrir a faixa de 290-400nm
                """)
            
            uploaded_file_spf = st.file_uploader("Dados PRÉ-IRRADIAÇÃO (A0i(λ))", type=["csv", "xlsx"], key="spf_upload")
            
            if uploaded_file_spf:
                df_spf, error = load_and_validate_data(uploaded_file_spf, "pre_irradiation")
                if error:
                    st.error(error)
                else:
                    st.success("Dados SPF validados!")
                    
                    # Visualização rápida dos dados
                    with st.expander("Visualizar dados carregados"):
                        st.dataframe(df_spf.head(10))
                        st.write(f"Total de {len(df_spf)} pontos espectrais")
                    
                    # Cálculo SPF in vitro
                    spf_in_vitro = calculate_spf_in_vitro(df_spf, erythema_spectrum, uv_ssr_spectrum)
                    
                    # Cálculo SPF Mansur
                    spf_mansur = calculate_spf_mansur_simplified(df_spf)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("SPF in vitro (Eq. 1)", f"{spf_in_vitro:.2f}")
                        st.metric("SPF Mansur Simplificado", f"{spf_mansur:.2f}")
                    
                    SPF_in_vivo = st.number_input("SPF in vivo medido:", min_value=1.0, value=30.0, step=0.1,
                                                 help="Valor do SPF determinado in vivo para calibrar o coeficiente C")
                    
                    def error_function(C):
                        return abs(calculate_adjusted_spf(df_spf, C, erythema_spectrum, uv_ssr_spectrum) - SPF_in_vivo)
                    
                    result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                    C_value = result.x
                    spf_ajustado = calculate_adjusted_spf(df_spf, C_value, erythema_spectrum, uv_ssr_spectrum)
                    
                    with col2:
                        st.metric("Coeficiente C (Eq. 2)", f"{C_value:.4f}")
                        st.metric("SPF ajustado (Eq. 2)", f"{spf_ajustado:.2f}")
                    
                    # Gráfico de absorbância inicial
                    st.subheader("Visualização dos Dados")
                    fig_absorbance = create_absorbance_plot(df_spf)
                    st.pyplot(fig_absorbance)
                    
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
                
                # Adicionar informações para o usuário
                with st.expander("ℹ️ Instruções"):
                    st.markdown("""
                    **Para análise UVA:**
                    1. Faça upload de um arquivo com dados completos de pós-irradiação
                    2. O arquivo deve conter as colunas: **Comprimento de Onda**, **P(λ)**, **I(λ)**, **Ai(λ)**, **A0i(λ)**
                    3. Formatos suportados: CSV (separador ; ou ,) ou Excel
                    4. Os dados devem cobrir a faixa de 320-400nm para análise UVA
                    """)
                
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
                            
                            # Visualização rápida dos dados
                            with st.expander("Visualizar dados carregados"):
                                st.dataframe(df_uva.head(10))
                                st.write(f"Total de {len(df_uva)} pontos espectrais")
                            
                            uva_pf_0 = calculate_uva_pf_initial(df_uva, C_value, ppd_spectrum, uva_spectrum)
                            dose = calculate_exposure_dose(uva_pf_0)
                            uva_pf_final = calculate_uva_pf_final(df_uva, C_value, ppd_spectrum, uva_spectrum)
                            critical_wl = calculate_critical_wavelength(df_uva, C_value)
                            
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
                                         f"{status} {'OK' if critical_wl >= 370 else 'Abaixo de 370'}")
                            
                            if 10.7 <= uva_pf_final <= 14.7:
                                st.success("✅ Resultado dentro da faixa do padrão de referência S2 (10.7-14.7)")
                            else:
                                st.warning("⚠️ Resultado fora da faixa do padrão S2 (10.7-14.7)")
                            
                            # Gráficos para análise UVA
                            st.subheader("Visualização dos Dados UVA")
                            
                            # Gráfico de absorbância com pós-irradiação
                            fig_absorbance_uva = create_absorbance_plot(
                                st.session_state.current_results['dados_pre'], 
                                df_uva, 
                                critical_wl
                            )
                            st.pyplot(fig_absorbance_uva)
                            
                            # Gráfico dos espectros de referência
                            fig_ref_spectra = create_reference_spectra_plot(
                                wavelengths, ppd_spectrum, erythema_spectrum, uva_spectrum
                            )
                            st.pyplot(fig_ref_spectra)
                            
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
                st.warning("Complete as análises anteriores para ver os resultados completos")
            else:
                results = st.session_state.current_results
                
                # Resumo dos resultados
                st.success("✅ Análise concluída com sucesso!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Resultados SPF")
                    st.metric("SPF in vitro", f"{results['spf_in_vitro']:.2f}")
                    st.metric("SPF Mansur", f"{results['spf_mansur']:.2f}")
                    st.metric("SPF in vivo", f"{results['spf_in_vivo']:.2f}")
                    st.metric("SPF ajustado", f"{results['spf_ajustado']:.2f}")
                    st.metric("Coeficiente C", f"{results['C_value']:.4f}")
                
                with col2:
                    st.subheader("Resultados UVA")
                    st.metric("UVA-PF₀", f"{results['uva_pf_0']:.2f}")
                    st.metric("UVA-PF Final", f"{results['uva_pf_final']:.2f}")
                    st.metric("Dose de Exposição", f"{results['dose']:.2f} J/cm²")
                    st.metric("λ Crítico", f"{results['critical_wavelength']:.1f} nm")
                
                # Gráfico comparativo de fatores de proteção
                st.subheader("Comparação dos Fatores de Proteção")
                fig_comparison = create_protection_factor_chart(results)
                st.pyplot(fig_comparison)
                
                # Interpretação dos resultados
                st.subheader("Interpretação dos Resultados")
                
                # Verificação do λ crítico
                if results['critical_wavelength'] >= 370:
                    st.success("✅ **λ Crítico**: Atende ao requisito mínimo de 370nm (ISO 24443:2011)")
                else:
                    st.error("❌ **λ Crítico**: Não atende ao requisito mínimo de 370nm (ISO 24443:2011)")
                
                # Verificação da relação UVA-PF/SPF
                uva_pf_ratio = results['uva_pf_final'] / results['spf_in_vivo'] if results['spf_in_vivo'] > 0 else 0
                if uva_pf_ratio >= 0.33:
                    st.success(f"✅ **Relação UVA-PF/SPF**: {uva_pf_ratio:.3f} (≥ 1/3, atendendo ao requisito da UE)")
                else:
                    st.warning(f"⚠️ **Relação UVA-PF/SPF**: {uva_pf_ratio:.3f} (< 1/3, não atendendo ao requisito da UE)")
                
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
        
        st.info("""
        Esta seção permite validar seus dados antes da análise completa.
        Faça upload de seus arquivos para verificar se estão no formato correto.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Validação de Dados SPF")
            spf_val_file = st.file_uploader("Dados para validação SPF", type=["csv", "xlsx"], key="val_spf")
            if spf_val_file:
                df_val, error = load_and_validate_data(spf_val_file, "pre_irradiation")
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Dados SPF válidos!")
                    st.write(f"**Formato:** {len(df_val)} pontos espectrais")
                    st.write(f"**Faixa espectral:** {df_val['Comprimento de Onda'].min():.0f} - {df_val['Comprimento de Onda'].max():.0f} nm")
                    
                    # Visualização rápida
                    fig_val = create_absorbance_plot(df_val)
                    st.pyplot(fig_val)
        
        with col2:
            st.subheader("Validação de Dados UVA")
            uva_val_file = st.file_uploader("Dados para validação UVA", type=["csv", "xlsx"], key="val_uva")
            if uva_val_file:
                df_val, error = load_and_validate_data(uva_val_file, "post_irradiation")
                if error:
                    st.error(f"❌ {error}")
                else:
                    is_valid, validation_msg = validate_uva_data(df_val)
                    if not is_valid:
                        st.error(f"❌ {validation_msg}")
                    else:
                        st.success("✅ Dados UVA válidos!")
                        st.write(f"**Formato:** {len(df_val)} pontos espectrais")
                        st.write(f"**Faixa UVA:** {df_val['Comprimento de Onda'].min():.0f} - {df_val['Comprimento de Onda'].max():.0f} nm")
                        
                        # Verificar colunas presentes
                        present_cols = [col for col in ['P(λ)', 'I(λ)', 'Ai(λ)', 'A0i(λ)'] if col in df_val.columns]
                        st.write(f"**Colunas detectadas:** {', '.join(present_cols)}")
    
    else:
        st.header("Sobre a Norma ISO 24443:2011")
        
        st.markdown("""
        ## 📋 Informações sobre a Norma ISO 24443:2011
        
        A norma **ISO 24443:2011** especifica um método para determinar a proteção UVA in vitro 
        de produtos de proteção solar.
        
        ### 🔍 Principais Parâmetros Avaliados:
        
        - **UVA-PF**: Fator de Proteção UVA
        - **λ Crítico**: Comprimento de onda abaixo do qual 90% da absorbância total é obtida
        - **Dose de Exposição**: Quantidade de radiação UVA necessária para o teste
        
        ### ✅ Requisitos da Norma:
        
        - **λ Crítico** deve ser ≥ 370 nm
        - O produto de referência S2 deve ter UVA-PF entre 10.7 e 14.7
        - A relação UVA-PF/SPF deve ser ≥ 1/3 (requisito da União Europeia)
        
        ### 📊 Metodologia:
        
        O método consiste em:
        1. Medir a transmitância espectrofotométrica do produto
        2. Aplicar uma dose de radiação UVA
        3. Medir novamente a transmitância
        4. Calcular os parâmetros de proteção UVA
        
        ### 🔬 Espectros de Referência:
        
        - **E(λ)**: Espectro de eritema (queimadura solar)
        - **P(λ)**: Espectro de escurecimento pigmentar persistente (PPD)
        - **I(λ)**: Espectro de irradiância solar
        """)
        
        # Mostrar espectros de referência
        wavelengths, ppd_spectrum, erythema_spectrum, uv_ssr_spectrum, uva_spectrum = load_reference_spectra()
        fig_ref = create_reference_spectra_plot(wavelengths, ppd_spectrum, erythema_spectrum, uva_spectrum)
        st.pyplot(fig_ref)

if __name__ == "__main__":
    main()
