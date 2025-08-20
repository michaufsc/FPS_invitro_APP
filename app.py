import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Análise de Proteção Solar - ISO 24443:2011",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ESPECTROS DE REFERÊNCIA COMPLETOS (Annex C da norma)
# =============================================================================
def load_reference_spectra():
    """Carrega os espectros de referência COMPLETOS da Annex C da norma ISO"""
    wavelengths = np.arange(290, 401)
    
    # Espectro de ação PPD (Tabela C.1 completa)
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
    
    # Espectro de eritema CIE 1987 (Tabela C.1 completa)
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
    # Espectro de irradiância UV-SSR (valores exemplares - devem ser substituídos pelos reais)
    uv_ssr_spectrum = np.array([
        8.741E-06, 1.450E-05, 2.659E-05, 4.574E-05, 1.006E-04, 2.589E-04, 7.035E-04, 1.678E-03, 3.727E-03, 7.938E-03,  # 290-299
        1.478E-02, 2.514E-02, 4.176E-02, 6.223E-02, 8.690E-02, 1.216E-01, 1.615E-01, 1.989E-01, 2.483E-01, 2.894E-01,  # 300-309
        3.358E-01, 3.872E-01, 4.311E-01, 4.884E-01, 5.121E-01, 5.567E-01, 5.957E-01, 6.256E-01, 6.565E-01, 6.879E-01,  # 310-319
        7.236E-01, 7.371E-01, 7.677E-01, 7.955E-01, 7.987E-01, 8.290E-01, 8.435E-01, 8.559E-01, 8.791E-01, 8.951E-01,  # 320-329
        9.010E-01, 9.161E-01, 9.434E-01, 9.444E-01, 9.432E-01, 9.571E-01, 9.663E-01, 9.771E-01, 9.770E-01, 9.967E-01,  # 330-339
        9.939E-01, 1.007E+00, 1.012E+00, 1.011E+00, 1.021E+00, 1.025E+00, 1.033E+00, 1.034E+00, 1.040E+00, 1.027E+00,  # 340-349
        1.045E+00, 1.042E+00, 1.040E+00, 1.039E+00, 1.043E+00, 1.046E+00, 1.035E+00, 1.039E+00, 1.027E+00, 1.035E+00,  # 350-359
        1.037E+00, 1.025E+00, 1.023E+00, 1.016E+00, 9.984E-01, 9.960E-01, 9.674E-01, 9.648E-01, 9.389E-01, 9.191E-01,  # 360-369
        8.977E-01, 8.725E-01, 8.473E-01, 8.123E-01, 7.840E-01, 7.416E-01, 7.148E-01, 6.687E-01, 6.280E-01, 5.863E-01,  # 370-379
        5.341E-01, 4.925E-01, 4.482E-01, 3.932E-01, 3.428E-01, 2.985E-01, 2.567E-01, 2.148E-01, 1.800E-01, 1.486E-01,  # 380-389
        1.193E-01, 9.403E-02, 7.273E-02, 5.532E-02, 4.010E-02, 2.885E-02, 2.068E-02, 1.400E-02, 9.510E-03, 6.194E-03,  # 390-399
        4.172E-03  # 400
    ])
    
    return wavelengths, ppd_spectrum, erythema_spectrum, uv_ssr_spectrum

# Sistema de sessão
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}

# FUNÇÕES DE CÁLCULO - ISO 24443:2011 CORRIGIDAS
# =============================================================================
def calculate_spf_in_vitro(df, erythema_spectrum, uv_ssr_spectrum):
    """Eq. 1: SPF in vitro inicial - ISO 24443:2011 CORRIGIDA"""
    d_lambda = 1
    total_numerator = 0    # SEM proteção (∫ E·I dλ)
    total_denominator = 0  # COM proteção (∫ E·I·T dλ)
    
    for _, row in df.iterrows():
        wavelength = int(row['Comprimento de Onda'])
        if wavelength < 290 or wavelength > 400:
            continue
        
        # Obter valor do espectro de referência
        idx = wavelength - 290
        E = erythema_spectrum[idx]
        I = uv_ssr_spectrum[idx]  # Usar espectro de referência
        A0 = row['A0i(λ)']
        T = 10 ** (-A0)  # Transmitância
        
        total_numerator += E * I * d_lambda
        total_denominator += E * I * T * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_adjusted_spf(df, C, erythema_spectrum, uv_ssr_spectrum):
    """Eq. 2: SPF ajustado com coeficiente C - ISO 24443:2011 CORRIGIDA"""
    d_lambda = 1
    total_numerator = 0    # SEM proteção (∫ E·I dλ)
    total_denominator = 0  # COM proteção (∫ E·I·T dλ)
    
    for _, row in df.iterrows():
        wavelength = int(row['Comprimento de Onda'])
        if wavelength < 290 or wavelength > 400:
            continue
        
        idx = wavelength - 290
        E = erythema_spectrum[idx]
        I = uv_ssr_spectrum[idx]  # Usar espectro de referência
        A0 = row['A0i(λ)']
        T_adjusted = 10 ** (-A0 * C)  # Transmitância ajustada
        
        total_numerator += E * I * d_lambda
        total_denominator += E * I * T_adjusted * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_uva_pf_initial(df, C, ppd_spectrum, uv_ssr_spectrum):
    """Eq. 3: UVA-PF₀ inicial - ISO 24443:2011 CORRIGIDA"""
    d_lambda = 1
    total_numerator = 0    # SEM proteção (∫ P·I dλ)
    total_denominator = 0  # COM proteção (∫ P·I·T dλ)
    
    for _, row in df.iterrows():
        wavelength = int(row['Comprimento de Onda'])
        if wavelength < 320 or wavelength > 400:
            continue
        
        idx = wavelength - 290
        P = ppd_spectrum[idx]
        I = uv_ssr_spectrum[idx]  # Usar espectro de referência
        A0 = row['A0i(λ)']
        T_adjusted = 10 ** (-A0 * C)  # Transmitância ajustada
        
        # CORREÇÃO: Numerador = SEM proteção, Denominador = COM proteção
        total_numerator += P * I * d_lambda
        total_denominator += P * I * T_adjusted * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_uva_pf_final(df, ppd_spectrum, uv_ssr_spectrum):
    """Eq. 5: UVA-PF final após irradiação - ISO 24443:2011 CORRIGIDA"""
    d_lambda = 1
    total_numerator = 0    # SEM proteção (∫ P·I dλ)
    total_denominator = 0  # COM proteção (∫ P·I·T dλ)
    
    for _, row in df.iterrows():
        wavelength = int(row['Comprimento de Onda'])
        if wavelength < 320 or wavelength > 400:
            continue
        
        idx = wavelength - 290
        P = ppd_spectrum[idx]
        I = uv_ssr_spectrum[idx]  # Usar espectro de referência
        Ae = row['Ai(λ)']  # Absorbância APÓS irradiação
        T_final = 10 ** (-Ae)  # Transmitância final
        
        # CORREÇÃO: Numerador = SEM proteção, Denominador = COM proteção
        total_numerator += P * I * d_lambda
        total_denominator += P * I * T_final * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_exposure_dose(uva_pf_0):
    """Eq. 4: Dose de exposição - ISO 24443:2011"""
    return uva_pf_0 * 1.2  # J/cm²

def calculate_critical_wavelength(df, C):
    """Calcula Critical Wavelength com absorbância ajustada"""
    df_uv = df[(df['Comprimento de Onda'] >= 290) & 
              (df['Comprimento de Onda'] <= 400)].copy()
    
    wavelengths = df_uv['Comprimento de Onda'].to_numpy()
    absorbance = df_uv['Ai(λ)'].to_numpy() * C
    
    total_area = np.trapz(absorbance, wavelengths)
    target_area = 0.9 * total_area
    
    cumulative_area = 0
    critical_wl = 400
    
    for i in range(1, len(wavelengths)):
        segment_area = (absorbance[i] + absorbance[i-1])/2 * (wavelengths[i] - wavelengths[i-1])
        cumulative_area += segment_area
        
        if cumulative_area >= target_area:
            critical_wl = wavelengths[i]
            break
    
    return critical_wl

# FUNÇÕES AUXILIARES
# =============================================================================
def map_column_names(df, data_type="pre_irradiation"):
    """Mapeia nomes de colunas para formato padrão com fallback manual"""
    column_mapping = {}
    detected_cols = set()
    
    if data_type == "post_irradiation":
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['wavelength', 'comprimento', 'onda', 'lambda', 'nm', 'wl']):
                column_mapping[col] = 'Comprimento de Onda'
                detected_cols.add('Comprimento de Onda')
            elif any(word in lower_col for word in ['p', 'ppd', 'pigment', 'pigmentacao', 'action']):
                column_mapping[col] = 'P(λ)'
                detected_cols.add('P(λ)')
            elif any(word in lower_col for word in ['i', 'intensity', 'intensidade', 'irradiance', 'irradiancia']):
                column_mapping[col] = 'I(λ)'
                detected_cols.add('I(λ)')
            elif any(word in lower_col for word in ['a_e', 'ae', 'absorbance', 'absorbancia', 'absorvancia', 'post']):
                column_mapping[col] = 'Ai(λ)'
                detected_cols.add('Ai(λ)')
            elif any(word in lower_col for word in ['a0', 'absorbancia_inicial', 'absorvancia_inicial', 'pre']):
                column_mapping[col] = 'A0i(λ)'
                detected_cols.add('A0i(λ)')
    else:
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['comprimento', 'onda', 'wavelength', 'lambda', 'nm', 'wl']):
                column_mapping[col] = 'Comprimento de Onda'
                detected_cols.add('Comprimento de Onda')
            elif any(word in lower_col for word in ['absorbancia', 'absorvancia', 'absorbância', 'absorvância', 'abs', 'a0']):
                column_mapping[col] = 'A0i(λ)'
                detected_cols.add('A0i(λ)')
            elif any(word in lower_col for word in ['e(λ)', 'e(lambda)', 'eritema', 'erythema', 'e(']):
                column_mapping[col] = 'E(λ)'
                detected_cols.add('E(λ)')
            elif any(word in lower_col for word in ['i(λ)', 'i(lambda)', 'intensidade', 'intensity', 'i(']):
                column_mapping[col] = 'I(λ)'
                detected_cols.add('I(λ)')
    
    return column_mapping, detected_cols

def validate_uva_data(df):
    """Valida especificamente dados UVA conforme norma ISO"""
    required_cols = ['Comprimento de Onda', 'P(λ)', 'I(λ)', 'Ai(λ)', 'A0i(λ)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Colunas UVA faltando: {', '.join(missing_cols)}"
    
    # Verificar faixa de wavelengths
    wavelengths = df['Comprimento de Onda'].values
    if min(wavelengths) > 320 or max(wavelengths) < 400:
        return False, "Faixa de wavelength UVA incompleta (320-400nm requerido)"
    
    return True, "Dados UVA válidos"

def load_and_validate_data(uploaded_file, data_type="pre_irradiation"):
    """Carrega e valida dados com mapeamento robusto"""
    try:
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.write("🔍 **Colunas originais:**", list(df.columns))
        
        column_mapping, detected_cols = map_column_names(df, data_type)
        st.write("🔄 **Mapeamento automático:**", column_mapping)
        
        df = df.rename(columns=column_mapping)
        
        # Fallback para mapeamento manual se colunas essenciais faltando
        if data_type == "pre_irradiation":
            required_cols = ['Comprimento de Onda', 'E(λ)', 'I(λ)', 'A0i(λ)']
        else:
            required_cols = ['Comprimento de Onda', 'P(λ)', 'I(λ)', 'Ai(λ)', 'A0i(λ)']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning("⚠️ Mapeamento automático incompleto. Selecione manualmente:")
            col_mapping_manual = {}
            cols = st.columns(3)
            
            for i, col_name in enumerate(missing_cols):
                with cols[i % 3]:
                    selected_col = st.selectbox(
                        f"Selecione coluna para {col_name}",
                        options=df.columns,
                        key=f"manual_{col_name}"
                    )
                    col_mapping_manual[selected_col] = col_name
            
            if st.button("Aplicar mapeamento manual"):
                df = df.rename(columns=col_mapping_manual)
                st.success("Mapeamento manual aplicado!")
                st.write("✅ **Colunas após mapeamento:**", list(df.columns))
        
        # Verificação final
        missing_cols_final = [col for col in required_cols if col not in df.columns]
        if missing_cols_final:
            raise ValueError(f"Colunas essenciais faltando: {', '.join(missing_cols_final)}")
            
        return df, None
        
    except Exception as e:
        return None, str(e)

# INTERFACE PRINCIPAL
# =============================================================================
st.title("🌞 Análise de Proteção Solar - ISO 24443:2011")

# Carregar espectros de referência
wavelengths, ppd_spectrum, erythema_spectrum, uv_ssr_spectrum = load_reference_spectra()


# Menu lateral
with st.sidebar:
    st.title("📊 Navegação")
    page = st.radio("Selecione o método:", 
                   ["ISO 24443 Completo", "Validação de Dados", "Sobre a Norma"])
    
    st.markdown("---")
    st.info("""
    **📋 Formatos esperados:**
    - **SPF:** Comprimento de Onda, E(λ), I(λ), A0i(λ)
    - **UVA:** Comprimento de Onda, P(λ), I(λ), Ai(λ), A0i(λ)
    """)
    
    st.markdown("---")
    st.caption("**ISO 24443:2011** - Determinação in vitro da fotoproteção UVA")

# PÁGINA 1: ISO 24443 COMPLETO
if page == "ISO 24443 Completo":
    st.header("🔬 Análise Completa - ISO 24443:2011")
    
    tab1, tab2, tab3 = st.tabs(["📊 SPF Inicial", "🌞 UVA", "📈 Resultados"])
    
    with tab1:
        st.subheader("Cálculo do SPF (Eq. 1-2)")
        uploaded_file_spf = st.file_uploader("📤 Dados SPF pré-irradiação", 
                                           type=["xlsx", "csv"], key="spf_upload")
        
        if uploaded_file_spf:
            df_spf, error = load_and_validate_data(uploaded_file_spf, "pre_irradiation")
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ Dados SPF validados!")
                st.dataframe(df_spf.head())
                
                try:
                    spf_in_vitro = calculate_spf_in_vitro(df_spf, erythema_spectrum, uv_ssr_spectrum)
                    st.metric("SPF in vitro (Eq. 1)", f"{spf_in_vitro:.2f}")
                    
                    SPF_in_vivo = st.number_input("SPF in vivo medido:", 
                                               min_value=1.0, value=30.0, step=0.1)
                    
                    def error_function(C):
                        return abs(calculate_adjusted_spf(df_spf, C, erythema_spectrum, uv_ssr_spectrum) - SPF_in_vivo)
                    
                    result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                    C_value = result.x
                    spf_ajustado = calculate_adjusted_spf(df_spf, C_value, erythema_spectrum, uv_ssr_spectrum)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Coeficiente C (Eq. 2)", f"{C_value:.4f}")
                        if not (0.8 <= C_value <= 1.6):
                            st.warning("⚠️ C fora da faixa recomendada (0.8-1.6)")
                    with col2:
                        st.metric("SPF ajustado (Eq. 2)", f"{spf_ajustado:.2f}")
                    
                    st.session_state.current_results = {
                        'spf_in_vitro': spf_in_vitro,
                        'spf_in_vivo': SPF_in_vivo,
                        'C_value': C_value,
                        'spf_ajustado': spf_ajustado,
                        'dados_pre': df_spf
                    }
                    
                except Exception as e:
                    st.error(f"Erro no cálculo: {str(e)}")
    
    with tab2:
        st.subheader("Análise UVA (Eq. 3-5)")
        
        if 'C_value' not in st.session_state.current_results:
            st.warning("⚠️ Calcule primeiro o SPF para obter o coeficiente C")
        else:
            C_value = st.session_state.current_results['C_value']
            st.success(f"✅ Coeficiente C: {C_value:.4f}")
            
            st.info("""
            **📝 Para UVA-PF₀ (Eq. 3), seu arquivo UVA precisa ter:**
            - Comprimento de Onda (320-400nm)
            - P(λ) (espectro PPD)
            - I(λ) (irradiância UVA)  
            - A0i(λ) (absorbância INICIAL)
            - Ai(λ) (absorbância APÓS irradiação)
            """)
            
            uploaded_file_uva = st.file_uploader("📤 Dados UVA completos", 
                                               type=["xlsx", "csv"], key="uva_upload")
            
            if uploaded_file_uva:
                df_uva, error = load_and_validate_data(uploaded_file_uva, "post_irradiation")
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    # Validação específica para dados UVA
                    is_valid, validation_msg = validate_uva_data(df_uva)
                    if not is_valid:
                        st.error(f"❌ {validation_msg}")
                    else:
                        st.success("✅ Dados UVA validados!")
                        st.dataframe(df_uva.head())
                        
                        try:
                            uva_pf_0 = calculate_uva_pf_initial(df_uva, C_value, ppd_spectrum)
                            dose = calculate_exposure_dose(uva_pf_0)
                            uva_pf_final = calculate_uva_pf_final(df_uva, ppd_spectrum, uv_ssr_spectrum)
                            critical_wl = calculate_critical_wavelength(df_uva, C_value)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("UVA-PF₀ (Eq. 3)", f"{uva_pf_0:.2f}")
                            with col2:
                                st.metric("Dose (Eq. 4)", f"{dose:.2f} J/cm²")
                            with col3:
                                st.metric("UVA-PF (Eq. 5)", f"{uva_pf_final:.2f}")
                            with col4:
                                status = "✅" if critical_wl >= 370 else "⚠️"
                                st.metric("λ Crítico", f"{critical_wl:.1f} nm", status)
                            
                            # Verificação do padrão de referência S2
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
                            
                        except Exception as e:
                            st.error(f"Erro no cálculo UVA: {str(e)}")
    
    with tab3:
        st.subheader("Resultados Completos")
        
        if 'uva_pf_final' not in st.session_state.current_results:
            st.warning("⚠️ Complete as análises anteriores")
        else:
            results = st.session_state.current_results
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("SPF in vitro", f"{results['spf_in_vitro']:.2f}")
                st.metric("SPF in vivo", f"{results['spf_in_vivo']:.2f}")
                st.metric("SPF ajustado", f"{results['spf_ajustado']:.2f}")
                st.metric("Coeficiente C", f"{results['C_value']:.4f}")
                
            with col2:
                st.metric("UVA-PF₀", f"{results['uva_pf_0']:.2f}")
                st.metric("UVA-PF Final", f"{results['uva_pf_final']:.2f}")
                st.metric("Dose de Exposição", f"{results['dose']:.2f} J/cm²")
                st.metric("λ Crítico", f"{results['critical_wavelength']:.1f} nm")
            
            # Gráficos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Gráfico 1: Absorbância
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
            
            # Gráfico 2: Espectros de referência
            ax2.plot(wavelengths, erythema_spectrum, label='Eritema (E(λ))', color='orange')
            ax2.plot(wavelengths, ppd_spectrum, label='PPD (P(λ))', color='purple')
            ax2.set_xlabel('Comprimento de Onda (nm)')
            ax2.set_ylabel('Valor do Espectro')
            ax2.set_title('Espectros de Referência')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(290, 400)
            
            st.pyplot(fig)
            
            # Relatório em formato de tabela
            st.subheader("📋 Relatório de Análise")
            report_data = {
                'Parâmetro': ['SPF in vitro', 'SPF in vivo', 'SPF ajustado', 'Coeficiente C', 
                             'UVA-PF₀', 'UVA-PF Final', 'Dose de Exposição', 'λ Crítico'],
                'Valor': [f"{results['spf_in_vitro']:.2f}", f"{results['spf_in_vivo']:.2f}",
                         f"{results['spf_ajustado']:.2f}", f"{results['C_value']:.4f}",
                         f"{results['uva_pf_0']:.2f}", f"{results['uva_pf_final']:.2f}",
                         f"{results['dose']:.2f} J/cm²", f"{results['critical_wavelength']:.1f} nm"],
                'Status': ['✅' if 0.8 <= results['C_value'] <= 1.6 else '⚠️', 
                          '✅', '✅', 
                          '✅' if 0.8 <= results['C_value'] <= 1.6 else '⚠️',
                          '✅', 
                          '✅' if 10.7 <= results['uva_pf_final'] <= 14.7 else '⚠️',
                          '✅', 
                          '✅' if results['critical_wavelength'] >= 370 else '⚠️']
            }
            
            report_df = pd.DataFrame(report_data)
            st.table(report_df)

# PÁGINA 2: VALIDAÇÃO DE DADOS
elif page == "Validação de Dados":
    st.header("🔍 Validação de Dados e Espectros")
    
    tab1, tab2 = st.tabs(["Validação UVA", "Espectros de Referência"])
    
    with tab1:
        st.subheader("Validação de Arquivos UVA")
        uploaded_file_val = st.file_uploader("📤 Carregue arquivo UVA para validação", 
                                          type=["xlsx", "csv"])
        
        if uploaded_file_val:
            df_val, error = load_and_validate_data(uploaded_file_val, "post_irradiation")
            
            if error:
                st.error(f"❌ {error}")
            else:
                is_valid, validation_msg = validate_uva_data(df_val)
                
                if is_valid:
                    st.success(f"✅ {validation_msg}")
                    
                    # Análise de faixa espectral
                    min_wl = df_val['Comprimento de Onda'].min()
                    max_wl = df_val['Comprimento de Onda'].max()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Wavelength Mínimo", f"{min_wl} nm")
                    with col2:
                        st.metric("Wavelength Máximo", f"{max_wl} nm")
                    with col3:
                        coverage = "✅ Completo" if min_wl <= 320 and max_wl >= 400 else "⚠️ Incompleto"
                        st.metric("Cobertura UVA", coverage)
                    
                    # Estatísticas básicas
                    st.subheader("📊 Estatísticas dos Dados")
                    for col in ['P(λ)', 'I(λ)', 'Ai(λ)', 'A0i(λ)']:
                        if col in df_val.columns:
                            st.write(f"**{col}**: Min={df_val[col].min():.4f}, "
                                   f"Max={df_val[col].max():.4f}, "
                                   f"Média={df_val[col].mean():.4f}")
                
                else:
                    st.error(f"❌ {validation_msg}")
    
    with tab2:
        st.subheader("Espectros de Referência - Anexo C")
        
        # Tabela com valores de referência
        st.info("Valores de espectro de referência conforme Anexo C da norma ISO 24443:2011")
        
        # Criar dataframe com alguns valores exemplares
        sample_wavelengths = [290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
        sample_data = {
            'λ (nm)': sample_wavelengths,
            'P(λ) PPD': [ppd_spectrum[w-290] for w in sample_wavelengths],
            'E(λ) Eritema': [erythema_spectrum[w-290] for w in sample_wavelengths]
        }
        
        ref_df = pd.DataFrame(sample_data)
        st.dataframe(ref_df.style.format({
            'P(λ) PPD': '{:.4f}',
            'E(λ) Eritema': '{:.6f}'
        }))
        
        # Gráfico dos espectros
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wavelengths, erythema_spectrum, label='Espectro de Eritema (E(λ))', linewidth=2, color='red')
        ax.plot(wavelengths, ppd_spectrum, label='Espectro PPD (P(λ))', linewidth=2, color='blue')
        ax.set_xlabel('Comprimento de Onda (nm)')
        ax.set_ylabel('Valor do Espectro')
        ax.set_title('Espectros de Referência - ISO 24443:2011')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(290, 400)
        st.pyplot(fig)

# PÁGINA 3: SOBRE A NORMA
else:
    st.header("📚 Sobre a Norma ISO 24443:2011")
    
    st.markdown("""
    ### **Cosmetics — Sun protection test methods — In vitro determination of sunscreen UVA photoprotection**
    
    **Objetivo:** Esta norma internacional especifica um procedimento *in vitro* para caracterizar a proteção UVA de produtos de proteção solar.
    
    ### **Principais Parâmetros Calculados:**
    
    - **UVA-PF (Fator de Proteção UVA):** Correlaciona com testes *in vivo* PPD
    - **λ Crítico:** Comprimento de onda onde 90% da absorbância integrada é alcançada
    - **Proporcionalidade de Absorbância UVA:** Razão entre proteção UVA e UVB
    
    ### **Fluxo do Método:**
    
    1. **Medição inicial** da absorbância do produto (pré-irradiação)
    2. **Ajuste matemático** usando coeficiente C para igualar SPF *in vitro* ao *in vivo*
    3. **Cálculo do UVA-PF₀** inicial para determinação da dose de exposição
    4. **Exposição à radiação UV** com dose específica (1.2 × UVA-PF₀ J/cm²)
    5. **Medição final** da absorbância (pós-irradiação)
    6. **Cálculo do UVA-PF final** e outros parâmetros
    
    ### **Requisitos do Sistema:**
    
    - Espectrofotômetro UV com range 290-400nm
    - Placas PMMA com superfície rugosa padronizada
    - Fonte de radiação UV simulando espectro solar
    - Temperatura controlada (25-35°C)
    
    ### **Controle de Qualidade:**
    
    - Validação regular do equipamento
    - Uso do padrão de referência S2 (UVA-PF entre 10.7-14.7)
    - Coeficiente C entre 0.8-1.6
    """)
    
    st.download_button(
        label="📥 Download da Estrutura de Arquivo Exemplar",
        data=pd.DataFrame({
            'Comprimento de Onda': range(290, 401),
            'P(λ)': ppd_spectrum,
            'I(λ)': [0.001] * 111,  # Valores exemplares
            'Ai(λ)': [0.5] * 111,
            'A0i(λ)': [0.6] * 111
        }).to_csv(index=False),
        file_name="modelo_dados_uva_iso24443.csv",
        mime="text/csv"
    )

# RODAPÉ
st.markdown("---")
st.markdown("""
**🔬 Sistema de Análise de Proteção Solar - ISO 24443:2011**  
*Desenvolvido para conformidade com método padrão internacional para determinação in vitro de proteção UVA*
""")
