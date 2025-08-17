import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from PIL import Image
import os

# ===================== CONFIGURAÇÃO INICIAL =====================
st.set_page_config(layout="wide", page_title="Cálculo de Fotoproteção In Vitro")

# ===================== FUNÇÕES AUXILIARES =====================
def carregar_imagem(caminho, largura):
    """Carrega imagem com tratamento de erros"""
    try:
        if os.path.exists(caminho):
            return Image.open(caminho)
        return None
    except Exception as e:
        st.error(f"Erro ao carregar imagem: {str(e)}")
        return None

# ===================== INTERFACE =====================
# Cabeçalho com logos
col1, col2 = st.columns([1, 0.2])
logo_principal = carregar_imagem("logo.png", 200)
logo_parceiro = carregar_imagem("logo_parceiro.png", 100)

with col1:
    if logo_principal:
        st.image(logo_principal, width=200)
    else:
        st.title("🌞 Fotoproteção In Vitro")

with col2:
    if logo_parceiro:
        st.image(logo_parceiro, width=100)

# ===================== ABAS PRINCIPAIS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ISO 24443", 
    "Mansur (1986)", 
    "CWC", 
    "UVA-PF", 
    "UVA1/Ultra-Longo"
])

# ===================== ABA 1: ISO 24443 =====================
with tab1:
    st.header("📊 ISO 24443: SPF In Vitro")
    st.markdown(r"""
    **Fórmula básica:**
    $$
    SPF = \frac{\int_{290}^{400} E(\lambda) \cdot I(\lambda) \, d\lambda}{\int_{290}^{400} E(\lambda) \cdot I(\lambda) \cdot T(\lambda) \, d\lambda}
    $$
    Onde:
    - $T(\lambda) = 10^{-A(\lambda)}$ (Transmitância)
    - $E(\lambda)$: Espectro de ação eritematosa (CIE 1998)
    - $I(\lambda)$: Irradiância solar (ISO/COLIPA)
    """)

    arquivo = st.file_uploader("Upload do arquivo de dados", type=["csv", "xlsx"])
    
    if arquivo:
        try:
            df = pd.read_csv(arquivo) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo)
            df.columns = df.columns.str.strip()
            
            # Pré-processamento
            df['Absorbancia'] = pd.to_numeric(df['Absorbancia'].astype(str).str.replace(',', '.'), errors='coerce')
            df['Transmitancia'] = 10 ** (-df['Absorbancia'])
            d_lambda = df['Comprimento de Onda'].diff().mean()

            # Cálculo SPF
            numerador = np.trapz(df['E(λ)'] * df['I(λ)'], x=df['Comprimento de Onda'])
            denominador = np.trapz(df['E(λ)'] * df['I(λ)'] * df['Transmitancia'], x=df['Comprimento de Onda'])
            spf = numerador / denominador

            st.success(f"**SPF In Vitro:** {spf:.2f}")
            
            # Ajuste do coeficiente C
            st.markdown(r"""
            **Ajuste para SPF in vivo:**
            $$
            C = \arg\min \left| SPF_{in\ vitro}(C) - SPF_{rotulado} \right|
            $$
            """)
            
            spf_rotulado = st.number_input("SPF Rotulado In Vivo", min_value=1.0, value=30.0)
            
            def erro(C):
                denom_ajustado = np.trapz(df['E(λ)'] * df['I(λ)'] * 10**(-df['Absorbancia'] * C), 
                                        x=df['Comprimento de Onda'])
                return abs((numerador / denom_ajustado) - spf_rotulado)
            
            resultado = opt.minimize_scalar(erro, bounds=(0.5, 1.5), method='bounded')
            C_ajustado = resultado.x
            st.success(f"**Coeficiente C ajustado:** {C_ajustado:.4f}")

        except Exception as e:
            st.error(f"Erro: {str(e)}")

# ===================== ABA 2: MANSUR =====================
with tab2:
    st.header("⚡ Método de Mansur (1986)")
    st.markdown(r"""
    **Fórmula simplificada para UVB:**
    $$
    SPF = 10 \times \sum_{\lambda=290}^{320} EE(\lambda) \cdot I(\lambda) \cdot A(\lambda)
    $$
    """)
    
    if 'df_iso' in st.session_state:
        df = st.session_state.df_iso.copy()
        df_mansur = df[(df['Comprimento de Onda'] >= 290) & (df['Comprimento de Onda'] <= 320)]
        
        # Tabela EE*I (Mansur 1986)
        ee_i = {
            290: 0.0150 * 0.134, 295: 0.0817 * 0.134,
            300: 0.2874 * 0.135, 305: 0.3278 * 0.136,
            310: 0.1864 * 0.137, 315: 0.0839 * 0.138,
            320: 0.0180 * 0.139
        }
        
        df_mansur['EE_I'] = df_mansur['Comprimento de Onda'].map(ee_i)
        df_mansur['Produto'] = df_mansur['EE_I'] * df_mansur['Absorbancia']
        spf_mansur = 10 * df_mansur['Produto'].sum()
        
        st.success(f"**SPF (Mansur):** {spf_mansur:.2f}")

# ===================== ABA 3: CWC =====================
with tab3:
    st.header("📏 Comprimento de Onda Crítico (CWC)")
    st.markdown(r"""
    **Definição ISO 24443:**
    $$
    CWC = \lambda \ \text{onde} \ \frac{\int_{290}^{\lambda} A(\lambda) d\lambda}{\int_{290}^{400} A(\lambda) d\lambda} \geq 0.9
    $$
    """)
    
    if 'df_iso' in st.session_state:
        df = st.session_state.df_iso.copy()
        df_cwc = df[(df['Comprimento de Onda'] >= 290) & (df['Comprimento de Onda'] <= 400)]
        
        area_total = np.trapz(df_cwc['Absorbancia'], x=df_cwc['Comprimento de Onda'])
        area_cumulativa = np.cumsum(df_cwc['Absorbancia'] * np.gradient(df_cwc['Comprimento de Onda']))
        idx_cwc = np.where(area_cumulativa >= 0.9 * area_total)[0][0]
        cwc = df_cwc['Comprimento de Onda'].iloc[idx_cwc]
        
        st.success(f"**CWC:** {cwc:.1f} nm {'✅ (≥ 370 nm)' if cwc >= 370 else '⚠️ (< 370 nm)'}")

# ===================== ABA 4: UVA-PF =====================
with tab4:
    st.header("🟣 UVA Protection Factor")
    st.markdown(r"""
    **Fórmula ISO 24443:**
    $$
    UVA\!-\!PF = \frac{\int_{320}^{400} P(\lambda) \cdot I(\lambda) d\lambda}{\int_{320}^{400} P(\lambda) \cdot I(\lambda) \cdot 10^{-A(\lambda) \cdot C} d\lambda}
    $$
    """)
    
    if 'df_iso' in st.session_state and 'C_ajustado' in st.session_state:
        df = st.session_state.df_iso.copy()
        df_uva = df[(df['Comprimento de Onda'] >= 320) & (df['Comprimento de Onda'] <= 400)]
        C = st.session_state.C_ajustado
        
        numerador = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'], x=df_uva['Comprimento de Onda'])
        denominador = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'] * 10**(-df_uva['Absorbancia'] * C), 
                             x=df_uva['Comprimento de Onda'])
        uva_pf = numerador / denominador
        
        st.success(f"**UVA-PF:** {uva_pf:.2f}")

# ===================== ABA 5: UVA1/ULTRA-LONGO =====================
with tab5:
    st.header("🔮 UVA1 & Ultra-Longo UVA")
    st.markdown(r"""
    **Métricas avançadas:**
    - **UVA1-PF (340-400 nm):** 
    $$
    \frac{\int_{340}^{400} E(\lambda)I(\lambda)d\lambda}{\int_{340}^{400} E(\lambda)I(\lambda)10^{-A(\lambda)C}d\lambda}
    $$
    - **Absorbância média (370-400 nm):**
    $$
    \frac{1}{30}\sum_{\lambda=370}^{400} A(\lambda)
    $$
    """)
    
    if 'df_iso' in st.session_state and 'C_ajustado' in st.session_state:
        df = st.session_state.df_iso.copy()
        C = st.session_state.C_ajustado
        
        # UVA1-PF
        df_uva1 = df[(df['Comprimento de Onda'] >= 340) & (df['Comprimento de Onda'] <= 400)]
        num_uva1 = np.trapz(df_uva1['E(λ)'] * df_uva1['I(λ)'], x=df_uva1['Comprimento de Onda'])
        den_uva1 = np.trapz(df_uva1['E(λ)'] * df_uva1['I(λ)'] * 10**(-df_uva1['Absorbancia'] * C), 
                          x=df_uva1['Comprimento de Onda'])
        uva1_pf = num_uva1 / den_uva1
        
        # Ultra-Longo UVA
        df_ultra = df[df['Comprimento de Onda'] >= 370]
        absorbancia_media = df_ultra['Absorbancia'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("UVA1-PF (340-400 nm)", f"{uva1_pf:.2f}")
        with col2:
            st.metric("Absorbância Média (370-400 nm)", f"{absorbancia_media:.3f}",
                    "✅ ≥ 0.8" if absorbancia_media >= 0.8 else "⚠️ < 0.8")

# ===================== RODAPÉ =====================
st.markdown("---")
st.caption("""
**Referências:**  
- ISO 24443:2012 - Determination of sunscreen UVA photoprotection in vitro  
- Mansur et al. (1986) - Correlação in vitro/in vivo  
- COLIPA (2009) - Guideline for UVA protection testing  
""")
