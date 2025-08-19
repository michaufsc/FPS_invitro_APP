import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# LOGOS NO TOPO
col1, col2 = st.columns([1, 0.5])
with col1:
    st.image("logo.png", width=200)
with col2:
    st.image("logo_ufsc.png", width=200)

# TÍTULO PRINCIPAL
st.title("🌞 Análise Completa de Proteção Solar")

# Funções de cálculo
def calculate_spf(df):
    """Calcula SPF in vitro conforme Equação 1"""
    d_lambda = 1
    E = df['E(λ)'].to_numpy()
    I = df['I(λ)'].to_numpy()
    A0i = df['A0i(λ)'].to_numpy()
    T = 10 ** (-A0i)
    
    numerator = np.sum(E * I * d_lambda)
    denominator = np.sum(E * I * T * d_lambda)
    
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    return numerator / denominator

def calculate_adjusted_spf(df, C):
    """Calcula SPF ajustado conforme Equação 2"""
    d_lambda = 1
    E = df['E(λ)'].to_numpy()
    I = df['I(λ)'].to_numpy()
    A0i = df['A0i(λ)'].to_numpy()
    
    numerator = np.sum(E * I * d_lambda)
    denominator = np.sum(E * I * (10 ** (-A0i * C)) * d_lambda)
    
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    return numerator / denominator

def calculate_uva_pf(df, C):
    """Calcula UVA-PF conforme Equação 3"""
    d_lambda = 1
    P = df['P(λ)'].to_numpy()
    I = df['I(λ)'].to_numpy()
    A0i = df['A0i(λ)'].to_numpy()
    
    numerator = np.sum(P * I * d_lambda)
    denominator = np.sum(P * I * (10 ** (-A0i * C)) * d_lambda)
    
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    return numerator / denominator

def calculate_uva_pf_i(df_post_irrad, C):
    """Calcula UVAPF-I (340-400 nm) conforme Equação 5"""
    mask = (df_post_irrad['Comprimento de Onda'] >= 340) & (df_post_irrad['Comprimento de Onda'] <= 400)
    df_uva = df_post_irrad[mask].copy()
    
    P = df_uva['P(λ)'].to_numpy()
    I = df_uva['I(λ)'].to_numpy()
    Ai = df_uva['Ai(λ)'].to_numpy()
    dλ = 1
    
    numerator = np.sum(P * I * dλ)
    denominator = np.sum(P * I * 10**(-Ai * C) * dλ)
    
    return numerator / denominator if denominator != 0 else 0

def calculate_critical_wavelength(df_post):
    """Calcula Critical Wavelength conforme Equação 7"""
    df_uva = df_post[(df_post['Comprimento de Onda'] >= 290) & 
                    (df_post['Comprimento de Onda'] <= 400)].copy()
    
    wavelengths = df_uva['Comprimento de Onda'].to_numpy()
    absorbance = df_uva['Ai(λ)'].to_numpy()
    
    total_area = np.trapz(absorbance, wavelengths)
    target_area = 0.9 * total_area
    
    cumulative_area = 0
    for i, (wl, abs) in enumerate(zip(wavelengths, absorbance)):
        if i == 0:
            continue
        segment_area = (abs + absorbance[i-1])/2 * (wl - wavelengths[i-1])
        cumulative_area += segment_area
        
        if cumulative_area >= target_area:
            return wl
    
    return 400

def calculate_uva_uv_ratio(df_post):
    """Calcula UVA-I/UV ratio conforme Equação 8"""
    mask_uva = (df_post['Comprimento de Onda'] >= 340) & (df_post['Comprimento de Onda'] <= 400)
    uva_area = np.trapz(df_post[mask_uva]['Ai(λ)'], df_post[mask_uva]['Comprimento de Onda'])
    
    mask_uv = (df_post['Comprimento de Onda'] >= 290) & (df_post['Comprimento de Onda'] <= 400)
    uv_area = np.trapz(df_post[mask_uv]['Ai(λ)'], df_post[mask_uv]['Comprimento de Onda'])
    
    return (uva_area/60) / (uv_area/110)

# Criação das abas
tab1, tab2, tab3 = st.tabs(["Cálculo SPF", "Análise UVA-PF", "Métricas Avançadas"])

with tab1:
    st.header("🔍 Cálculo do Fator de Proteção Solar (SPF)")
    
    # Upload do arquivo
    uploaded_file = st.file_uploader("Carregue dados pré-irradiação (Excel/CSV)", 
                                   type=["xlsx", "csv"], 
                                   key="spf_upload")
    
    if uploaded_file:
        try:
            # Leitura dos dados
            if uploaded_file.name.endswith('xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            df.columns = df.columns.str.strip()
            
            # Verificação de colunas
            required_cols = ['Comprimento de Onda', 'E(λ)', 'I(λ)', 'A0i(λ)']
            if not all(col in df.columns for col in required_cols):
                missing = [c for c in required_cols if c not in df.columns]
                st.error(f"Colunas faltando: {', '.join(missing)}")
            else:
                # Cálculo do SPF
                try:
                    spf = calculate_spf(df)
                    st.success(f"📊 SPF in vitro calculado: {spf:.2f}")
                    
                    # Ajuste do SPF
                    st.subheader("🔧 Ajuste do SPF")
                    SPF_label = st.number_input("SPF rotulado (in vivo)", 
                                             min_value=1.0, value=30.0, step=0.1)
                    
                    # Otimização para encontrar C
                    def error_function(C):
                        return abs(calculate_adjusted_spf(df, C) - SPF_label)
                    
                    result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                    C_adjusted = result.x
                    SPF_adjusted = calculate_adjusted_spf(df, C_adjusted)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Coeficiente de Ajuste (C)", f"{C_adjusted:.4f}")
                    with col2:
                        st.metric("SPF Ajustado", f"{SPF_adjusted:.2f}")
                    
                    # Visualização
                    st.subheader("📈 Visualização dos Dados")
                    fig = px.line(df, x='Comprimento de Onda', y=['A0i(λ)', 'E(λ)', 'I(λ)'],
                                title="Dados Pré-Irradiação")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except ValueError as e:
                    st.error(f"Erro no cálculo: {str(e)}")
                
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

with tab2:
    st.header("🌞 Análise do Fator de Proteção UVA (UVA-PF)")
    
    # Upload dos dados pós-irradiação
    post_irrad_file = st.file_uploader("Carregue dados pós-irradiação (Excel/CSV)", 
                                     type=["xlsx", "csv"], 
                                     key="uva_pf_upload")
    
    if post_irrad_file:
        try:
            # Leitura dos dados
            if post_irrad_file.name.endswith('xlsx'):
                df_post = pd.read_excel(post_irrad_file)
            else:
                df_post = pd.read_csv(post_irrad_file)
            
            df_post.columns = df_post.columns.str.strip()
            
            # Verificação de colunas
            required_cols = ['Comprimento de Onda', 'P(λ)', 'I(λ)', 'Ai(λ)']
            if not all(col in df_post.columns for col in required_cols):
                missing = [c for c in required_cols if c not in df_post.columns]
                st.error(f"Colunas faltando: {', '.join(missing)}")
            else:
                # Coeficiente C e SPF rotulado
                C = st.number_input("Coeficiente de ajuste (C)", 
                                  min_value=0.1, max_value=2.0, 
                                  value=1.0, step=0.01,
                                  key="uva_coef")
                
                labelled_spf = st.number_input("SPF rotulado", 
                                             min_value=1.0, value=30.0,
                                             key="labelled_spf")
                
                # Cálculos UVA
                uva_pf = calculate_uva_pf(df_post, C)
                uva_pf_i = calculate_uva_pf_i(df_post, C)
                ratio = uva_pf_i / labelled_spf if labelled_spf != 0 else 0
                
                # Exibição dos resultados
                st.subheader("📊 Resultados UVA")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("UVA-PF", f"{uva_pf:.2f}")
                with col2:
                    st.metric("UVA-PF-I", f"{uva_pf_i:.2f}")
                with col3:
                    st.metric("UVA-PF-I/SPF Ratio", f"{ratio:.2f}",
                             "Bom (≥1/3)" if ratio >= (1/3) else "Abaixo do recomendado")
                
                # Visualização
                st.subheader("📈 Espectro de Absorbância Pós-Irradiação")
                fig = px.line(df_post, x='Comprimento de Onda', y='Ai(λ)',
                            title="Absorbância após Irradiação")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

with tab3:
    st.header("🔬 Métricas Avançadas de Proteção UVA")
    
    if 'df_post' in globals():
        # Critical Wavelength
        cw = calculate_critical_wavelength(df_post)
        
        # UVA-I/UV Ratio
        uva_uv_ratio = calculate_uva_uv_ratio(df_post)
        
        # Exibição dos resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Comprimento de Onda Crítico (nm)", f"{cw:.1f}",
                     "Bom (≥370 nm)" if cw >= 370 else "Abaixo do recomendado")
        with col2:
            st.metric("Razão UVA-I/UV", f"{uva_uv_ratio:.2f}")
        
        # Visualização avançada
        st.subheader("📊 Análise do Espectro UVA")
        
        # Criando área cumulativa
        df_uv = df_post[(df_post['Comprimento de Onda'] >= 290) & 
                       (df_post['Comprimento de Onda'] <= 400)].copy()
        df_uv['Cumulative Area'] = df_uv['Ai(λ)'].cumsum()
        total_area = df_uv['Cumulative Area'].max()
        
        fig = go.Figure()
        
        # Absorbância
        fig.add_trace(go.Scatter(
            x=df_uv['Comprimento de Onda'],
            y=df_uv['Ai(λ)'],
            name='Absorbância',
            yaxis='y1'
        ))
        
        # Área cumulativa
        fig.add_trace(go.Scatter(
            x=df_uv['Comprimento de Onda'],
            y=df_uv['Cumulative Area'],
            name='Área Cumulativa',
            yaxis='y2',
            line=dict(dash='dot')
        ))
        
        # Linha do CW
        fig.add_vline(x=cw, line_dash="dash", line_color="red",
                     annotation_text=f'CW: {cw:.1f} nm')
        
        fig.update_layout(
            title="Análise do Comprimento de Onda Crítico",
            xaxis_title="Comprimento de Onda (nm)",
            yaxis_title="Absorbância",
            yaxis2=dict(title="Área Cumulativa", overlaying='y', side='right'),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Por favor, carregue os dados pós-irradiação na aba 'Análise UVA-PF' primeiro")

# Rodapé
st.markdown("---")
st.markdown("""
**Referências:**  
- ISO 24443:2012 - Determination of sunscreen UVA photoprotection in vitro  
- COLIPA/CTFA/JCIA: International sun protection factor (SPF) test method  
- FDA: Broad Spectrum Test Procedure
""")
