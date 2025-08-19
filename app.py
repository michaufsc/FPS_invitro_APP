import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Análise de Proteção Solar",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sistema de sessão
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}

# FUNÇÕES DE CÁLCULO - ISO 24443:2012
# =============================================================================
def calculate_spf_in_vitro(df):
    """Eq. 1: SPF in vitro inicial - ISO 24443:2012"""
    d_lambda = 1
    total_numerator = 0
    total_denominator = 0
    
    for _, row in df.iterrows():
        wavelength = row['Comprimento de Onda']
        if wavelength < 290 or wavelength > 400:
            continue
            
        A0 = row['A0i(λ)']
        E = row['E(λ)']
        I = row['I(λ)']
        T = 10 ** (-A0)
        
        total_numerator += E * I * d_lambda
        total_denominator += E * I * T * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_adjusted_spf(df, C):
    """Eq. 2: SPF ajustado com coeficiente C - ISO 24443:2012"""
    d_lambda = 1
    total_numerator = 0
    total_denominator = 0
    
    for _, row in df.iterrows():
        wavelength = row['Comprimento de Onda']
        if wavelength < 290 or wavelength > 400:
            continue
            
        A0 = row['A0i(λ)']
        E = row['E(λ)']
        I = row['I(λ)']
        T_adjusted = 10 ** (-A0 * C)
        
        total_numerator += E * I * d_lambda
        total_denominator += E * I * T_adjusted * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_uva_pf_initial(df_uva, C):
    """Eq. 3: UVA-PF₀ inicial - ISO 24443:2012 - CORRIGIDA"""
    d_lambda = 1
    total_numerator = 0
    total_denominator = 0
    
    for _, row in df_uva.iterrows():
        wavelength = row['Comprimento de Onda']
        if wavelength < 320 or wavelength > 400:
            continue
            
        # PARA UVA-PF₀: Usamos absorbância INICIAL (A0i) mas do arquivo UVA
        # Isso assume que o arquivo UVA também tem coluna A0i(λ)
        A0 = row['A0i(λ)']  # Absorbância inicial
        P = row['P(λ)']     # Espectro PPD
        I = row['I(λ)']     # Irradiância UVA
        T_adjusted = 10 ** (-A0 * C)
        
        total_numerator += P * I * d_lambda
        total_denominator += P * I * T_adjusted * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_uva_pf_final(df_uva, C):
    """Eq. 5: UVA-PF final após irradiação - ISO 24443:2012"""
    d_lambda = 1
    total_numerator = 0
    total_denominator = 0
    
    for _, row in df_uva.iterrows():
        wavelength = row['Comprimento de Onda']
        if wavelength < 320 or wavelength > 400:
            continue
            
        Ae = row['Ai(λ)']  # Absorbância APÓS irradiação
        Af = Ae * C        # Absorbância ajustada: Af(λ) = Ae(λ) * C
        P = row['P(λ)']    # Espectro PPD
        I = row['I(λ)']    # Irradiância UVA
        T_final = 10 ** (-Af)
        
        total_numerator += P * I * d_lambda
        total_denominator += P * I * T_final * d_lambda
    
    return total_numerator / total_denominator if total_denominator != 0 else 0

def calculate_exposure_dose(uva_pf_0):
    """Eq. 4: Dose de exposição - ISO 24443:2012"""
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

# FUNÇÃO MANSUR (SIMPLIFICADA)
# =============================================================================
def calculate_spf_mansur(df, CF=10):
    """
    Calcula FPS usando método de Mansur (1986)
    SPF = CF × Σ[E(λ)×I(λ)] × Σ[A(λ)] / Σ[E(λ)×I(λ)×A(λ)]
    """
    d_lambda = 1
    sum_ei = 0
    sum_a = 0
    sum_eia = 0
    
    for _, row in df.iterrows():
        wavelength = row['Comprimento de Onda']
        if wavelength < 290 or wavelength > 400:
            continue
            
        A = row['A0i(λ)']
        E = row['E(λ)']
        I = row['I(λ)']
        
        sum_ei += E * I * d_lambda
        sum_a += A * d_lambda
        sum_eia += E * I * A * d_lambda
    
    if sum_eia == 0:
        return 0
    
    return CF * sum_ei * sum_a / sum_eia

# FUNÇÕES AUXILIARES
# =============================================================================
def map_column_names(df, data_type="pre_irradiation"):
    """Mapeia nomes de colunas para formato padrão"""
    column_mapping = {}
    
    if data_type == "post_irradiation":
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['wavelength', 'comprimento', 'onda', 'lambda', 'nm']):
                column_mapping[col] = 'Comprimento de Onda'
            elif any(word in lower_col for word in ['p', 'ppd', 'pigment', 'pigmentacao']):
                column_mapping[col] = 'P(λ)'
            elif any(word in lower_col for word in ['i', 'intensity', 'intensidade', 'irradiance']):
                column_mapping[col] = 'I(λ)'
            elif any(word in lower_col for word in ['a_e', 'ae', 'absorbance', 'absorbancia', 'absorvancia']):
                column_mapping[col] = 'Ai(λ)'
            elif any(word in lower_col for word in ['a0', 'absorbancia_inicial', 'absorvancia_inicial']):
                column_mapping[col] = 'A0i(λ)'
        
        # Garantir mapeamento por posição
        if len(column_mapping) < 4 and len(df.columns) >= 4:
            column_mapping = {
                df.columns[0]: 'Comprimento de Onda',
                df.columns[1]: 'P(λ)',
                df.columns[2]: 'I(λ)', 
                df.columns[3]: 'Ai(λ)'
            }
            
    else:
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['comprimento', 'onda', 'wavelength', 'lambda', 'nm']):
                column_mapping[col] = 'Comprimento de Onda'
            elif any(word in lower_col for word in ['absorbancia', 'absorvancia', 'absorbância', 'absorvância', 'abs', 'a0']):
                column_mapping[col] = 'A0i(λ)'
            elif any(word in lower_col for word in ['e(λ)', 'e(lambda)', 'eritema', 'erythema', 'e(']):
                column_mapping[col] = 'E(λ)'
            elif any(word in lower_col for word in ['i(λ)', 'i(lambda)', 'intensidade', 'intensity', 'i(']):
                column_mapping[col] = 'I(λ)'
    
    return column_mapping

def load_and_validate_data(uploaded_file, data_type="pre_irradiation"):
    """Carrega e valida dados com mapeamento"""
    try:
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.write("🔍 **Colunas originais:**", list(df.columns))
        
        column_mapping = map_column_names(df, data_type)
        st.write("🔄 **Mapeamento:**", column_mapping)
        
        df = df.rename(columns=column_mapping)
        st.write("✅ **Colunas após mapeamento:**", list(df.columns))
        
        if data_type == "pre_irradiation":
            required_cols = ['Comprimento de Onda', 'E(λ)', 'I(λ)', 'A0i(λ)']
        else:
            required_cols = ['Comprimento de Onda', 'P(λ)', 'I(λ)', 'Ai(λ)']
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Colunas faltando: {', '.join(missing_cols)}")
            
        return df, None
        
    except Exception as e:
        return None, str(e)

# INTERFACE PRINCIPAL
# =============================================================================
st.title("🌞 Análise de Proteção Solar")

# Menu lateral
with st.sidebar:
    st.title("📊 Navegação")
    page = st.radio("Selecione o método:", 
                   ["ISO 24443 Completo", "Método Mansur Simples"])
    
    st.markdown("---")
    st.info("""
    **📋 Formatos esperados:**
    - **SPF:** Comprimento de Onda, E(λ), I(λ), A0i(λ)
    - **UVA:** Comprimento de Onda, P(λ), I(λ), Ai(λ), A0i(λ)
    """)

# PÁGINA 1: ISO 24443 COMPLETO
if page == "ISO 24443 Completo":
    st.header("🔬 Análise Completa - ISO 24443:2012")
    
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
                    spf_in_vitro = calculate_spf_in_vitro(df_spf)
                    st.metric("SPF in vitro (Eq. 1)", f"{spf_in_vitro:.2f}")
                    
                    SPF_in_vivo = st.number_input("SPF in vivo medido:", 
                                               min_value=1.0, value=30.0, step=0.1)
                    
                    def error_function(C):
                        return abs(calculate_adjusted_spf(df_spf, C) - SPF_in_vivo)
                    
                    result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                    C_value = result.x
                    spf_ajustado = calculate_adjusted_spf(df_spf, C_value)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Coeficiente C (Eq. 2)", f"{C_value:.4f}")
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
                    st.error(f"Erro no cálculo: {e}")
    
    with tab2:
        st.subheader("Análise UVA (Eq. 3-5)")
        
        if 'C_value' not in st.session_state.current_results:
            st.warning("⚠️ Calcule primeiro o SPF")
        else:
            C_value = st.session_state.current_results['C_value']
            st.success(f"✅ Coeficiente C: {C_value:.4f}")
            
            st.info("""
            **📝 Para UVA-PF₀ (Eq. 3), seu arquivo UVA precisa ter:**
            - Comprimento de Onda
            - P(λ) (espectro PPD)
            - I(λ) (irradiância UVA)  
            - A0i(λ) (absorbância INICIAL - mesma do SPF)
            - Ai(λ) (absorbância APÓS irradiação)
            """)
            
            uploaded_file_uva = st.file_uploader("📤 Dados UVA completos", 
                                               type=["xlsx", "csv"], key="uva_upload")
            
            if uploaded_file_uva:
                df_uva, error = load_and_validate_data(uploaded_file_uva, "post_irradiation")
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Dados UVA validados!")
                    st.dataframe(df_uva.head())
                    
                    # Verificar se tem A0i(λ) para UVA-PF₀
                    if 'A0i(λ)' not in df_uva.columns:
                        st.error("❌ Arquivo UVA precisa da coluna A0i(λ) para cálculo do UVA-PF₀")
                    else:
                        try:
                            uva_pf_0 = calculate_uva_pf_initial(df_uva, C_value)
                            dose = calculate_exposure_dose(uva_pf_0)
                            uva_pf_final = calculate_uva_pf_final(df_uva, C_value)
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
                            
                            st.session_state.current_results.update({
                                'uva_pf_0': uva_pf_0,
                                'dose': dose,
                                'uva_pf_final': uva_pf_final,
                                'critical_wavelength': critical_wl,
                                'dados_post': df_uva
                            })
                            
                        except Exception as e:
                            st.error(f"Erro no cálculo UVA: {e}")
    
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
                
            with col2:
                st.metric("UVA-PF₀", f"{results['uva_pf_0']:.2f}")
                st.metric("UVA-PF Final", f"{results['uva_pf_final']:.2f}")
                st.metric("Dose", f"{results['dose']:.2f} J/cm²")
                st.metric("λ Crítico", f"{results['critical_wavelength']:.1f} nm")
            
            # Gráficos
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results['dados_pre']['Comprimento de Onda'], 
                   results['dados_pre']['A0i(λ)'], 
                   label='Absorbância Inicial (SPF)', linewidth=2, color='blue')
            
            if 'dados_post' in results:
                ax.plot(results['dados_post']['Comprimento de Onda'], 
                       results['dados_post']['Ai(λ)'], 
                       label='Absorbância Final (UVA)', linewidth=2, color='red')
            
            ax.set_xlabel('Comprimento de Onda (nm)')
            ax.set_ylabel('Absorbância')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# PÁGINA 2: MÉTODO MANSUR SIMPLES
else:
    st.header("🧪 Método Mansur Simplificado")
    
    uploaded_file = st.file_uploader("📤 Dados para cálculo Mansur", 
                                   type=["xlsx", "csv"], key="mansur_upload")
    
    if uploaded_file:
        df, error = load_and_validate_data(uploaded_file, "pre_irradiation")
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ Dados validados!")
            st.dataframe(df.head())
            
            try:
                spf_mansur = calculate_spf_mansur(df)
                
                st.metric("FPS (Método Mansur)", f"{spf_mansur:.2f}")
                
                # Gráfico simples
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['Comprimento de Onda'], df['A0i(λ)'], 
                       label='Absorbância', linewidth=2, color='green')
                ax.set_xlabel('Comprimento de Onda (nm)')
                ax.set_ylabel('Absorbância')
                ax.set_title('Espectro de Absorbância - Método Mansur')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.info("""
                **📝 Equação de Mansur:**
                `SPF = 10 × Σ[E(λ)×I(λ)] × Σ[A(λ)] / Σ[E(λ)×I(λ)×A(λ)]`
                """)
                
            except Exception as e:
                st.error(f"Erro no cálculo: {e}")

# RODAPÉ
st.markdown("---")
st.markdown("**🔬 Sistema de Análise de Proteção Solar**")
