import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# ===================== CONFIGURAÇÃO INICIAL =====================
st.set_page_config(layout="wide")
st.title("🧪 Calculadora de Fotoproteção *In Vitro*")

# ===================== CABEÇALHO =====================
col1, col2 = st.columns([1, 0.3])
with col1:
    st.image("logo.png", width=200)
with col2:
    st.image("logo_ufsc.png", width=100)

st.markdown("""
**Ferramenta para cálculo de SPF *in vitro*, UVA-PF, CWC e métricas UVA1/Ultra-Longo UVA**, conforme:
- ISO 24443 (2012)
- Método de Mansur (1986)
- COLIPA (2009)
""")

# ===================== ABAS PRINCIPAIS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ISO 24443 (Ajustado)", 
    "Mansur (1986)", 
    "CWC", 
    "UVA-PF", 
    "UVA1 & Ultra-Longo UVA"
])

# ===================== VARIÁVEIS GLOBAIS =====================
if "df_iso" not in st.session_state:
    st.session_state.df_iso = None
if "spf_iso" not in st.session_state:
    st.session_state.spf_iso = None
if "C_ajustado" not in st.session_state:
    st.session_state.C_ajustado = 0.8  # Valor padrão

# =====================================================
# ABA 1: ISO 24443 (SPF *In Vitro* Ajustado)
# =====================================================
with tab1:
    st.subheader("📊 ISO 24443: Cálculo do SPF *In Vitro*")
    st.markdown("""
    **Fórmula do SPF *in vitro* (sem ajuste):**
    \[
    SPF = \frac{\sum_{290}^{400} E(\lambda) \cdot I(\lambda) \cdot \Delta\lambda}{\sum_{290}^{400} E(\lambda) \cdot I(\lambda) \cdot T(\lambda) \cdot \Delta\lambda}
    \]
    - \(T(\lambda) = 10^{-A(\lambda)}\): Transmitância.
    - \(E(\lambda)\): Espectro de ação eritematosa (CIE 1998).
    - \(I(\lambda)\): Irradiância solar (ISO/COLIPA).
    - \(\Delta\lambda\): Passo entre comprimentos de onda (ex: 1 nm).
    """)

    uploaded_file = st.file_uploader("📤 Carregue o arquivo de absorbância (290–400 nm)", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            
            # Pré-processamento
            df['Absorbancia'] = pd.to_numeric(df['Absorbancia'].astype(str).str.replace(',', '.'), errors='coerce')
            df['Transmitancia'] = 10 ** (-df['Absorbancia'])
            d_lambda = df['Comprimento de Onda'].diff().mean()  # Passo Δλ automático

            # Cálculo do SPF *in vitro* (ISO 24443)
            numerador = np.trapz(df['E(λ)'] * df['I(λ)'], x=df['Comprimento de Onda'])
            denominador = np.trapz(df['E(λ)'] * df['I(λ)'] * df['Transmitancia'], x=df['Comprimento de Onda'])
            spf = numerador / denominador

            st.success(f"**SPF *In Vitro* (ISO 24443):** {spf:.2f}")
            st.session_state.df_iso = df
            st.session_state.spf_iso = spf

            # Gráfico de Transmitância
            fig, ax = plt.subplots()
            ax.plot(df['Comprimento de Onda'], df['Transmitancia'], color='blue')
            ax.set_xlabel("Comprimento de Onda (nm)")
            ax.set_ylabel("Transmitância")
            st.pyplot(fig)

            # Ajuste do SPF para *in vivo* (coeficiente C)
            st.markdown("""
            **Ajuste para SPF *in vivo*:**
            \[
            SPF_{\text{ajustado}} = \frac{\sum E \cdot I \cdot \Delta\lambda}{\sum E \cdot I \cdot 10^{-A \cdot C} \cdot \Delta\lambda}
            \]
            """)
            spf_label = st.number_input("SPF rotulado *in vivo*:", min_value=1.0, value=30.0)
            
            def error(C):
                denom_ajustado = np.trapz(df['E(λ)'] * df['I(λ)'] * 10**(-df['Absorbancia'] * C), x=df['Comprimento de Onda'])
                return abs((numerador / denom_ajustado) - spf_label)
            
            result = opt.minimize_scalar(error, bounds=(0.5, 1.5), method='bounded')
            C_ajustado = result.x
            st.session_state.C_ajustado = C_ajustado
            st.success(f"**Coeficiente C ajustado:** {C_ajustado:.4f}")

        except Exception as e:
            st.error(f"Erro: {e}")

# =====================================================
# ABA 2: Método de Mansur (1986) - SPF Rápido (UVB)
# =====================================================
with tab2:
    st.subheader("⚡ Método de Mansur (1986) - SPF Estimado (UVB)")
    st.markdown("""
    **Fórmula:**
    \[
    SPF = 10 \times \sum_{\lambda=290}^{320} EE(\lambda) \cdot I(\lambda) \cdot A(\lambda)
    \]
    - \(EE(\lambda) \cdot I(\lambda)\): Valores pré-definidos (Mansur et al., 1986).
    - \(A(\lambda)\): Absorbância da amostra.
    - **Atenção**: Só considera UVB (290–320 nm).
    """)

    if st.session_state.df_iso is not None:
        df_mansur = st.session_state.df_iso.copy()
        df_mansur = df_mansur[(df_mansur['Comprimento de Onda'] >= 290) & (df_mansur['Comprimento de Onda'] <= 320)]
        
        # Tabela de EE*I do Mansur (1986)
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
        st.dataframe(df_mansur[['Comprimento de Onda', 'Absorbancia', 'EE_I', 'Produto']].style.format("{:.4f}"))

# =====================================================
# ABA 3: Comprimento de Onda Crítico (CWC)
# =====================================================
with tab3:
    st.subheader("📏 Comprimento de Onda Crítico (CWC)")
    st.markdown("""
    **Definição (ISO 24443):**
    \[
    CWC = \lambda \text{ onde } \frac{\sum_{290}^{\lambda} A(\lambda) \Delta\lambda}{\sum_{290}^{400} A(\lambda) \Delta\lambda} \geq 0.9
    \]
    - **CWC ≥ 370 nm** indica proteção UVA ampla.
    """)

    if st.session_state.df_iso is not None:
        df_cwc = st.session_state.df_iso.copy()
        df_cwc = df_cwc[(df_cwc['Comprimento de Onda'] >= 290) & (df_cwc['Comprimento de Onda'] <= 400)]
        area_total = np.trapz(df_cwc['Absorbancia'], x=df_cwc['Comprimento de Onda'])
        area_cumulativa = np.cumsum(df_cwc['Absorbancia'] * np.gradient(df_cwc['Comprimento de Onda']))
        cwc_index = np.where(area_cumulativa >= 0.9 * area_total)[0][0]
        cwc = df_cwc['Comprimento de Onda'].iloc[cwc_index]

        st.success(f"**CWC:** {cwc:.1f} nm {'✅ (≥ 370 nm)' if cwc >= 370 else '⚠️ (< 370 nm)'}")

        # Gráfico
        fig, ax = plt.subplots()
        ax.plot(df_cwc['Comprimento de Onda'], df_cwc['Absorbancia'], label="Absorbância")
        ax.axvline(cwc, color='red', linestyle='--', label=f"CWC = {cwc:.1f} nm")
        ax.fill_between(df_cwc['Comprimento de Onda'], 0, df_cwc['Absorbancia'], alpha=0.2)
        ax.set_xlabel("Comprimento de Onda (nm)")
        ax.set_ylabel("Absorbância")
        ax.legend()
        st.pyplot(fig)

# =====================================================
# ABA 4: UVA-PF (Fator de Proteção UVA)
# =====================================================
with tab4:
    st.subheader("🟣 UVA-PF (320–400 nm)")
    st.markdown("""
    **Fórmula (ISO 24443):**
    \[
    UVA\!-\!PF = \frac{\sum_{320}^{400} P(\lambda) \cdot I(\lambda) \cdot \Delta\lambda}{\sum_{320}^{400} P(\lambda) \cdot I(\lambda) \cdot 10^{-A(\lambda) \cdot C} \cdot \Delta\lambda}
    \]
    - \(P(\lambda)\): Espectro de ação UVA (COLIPA 2009).
    - \(C\): Coeficiente ajustado da Aba 1.
    """)

    if st.session_state.df_iso is not None:
        df_uva = st.session_state.df_iso.copy()
        df_uva = df_uva[(df_uva['Comprimento de Onda'] >= 320) & (df_uva['Comprimento de Onda'] <= 400)]
        d_lambda = df_uva['Comprimento de Onda'].diff().mean()
        C = st.session_state.C_ajustado

        numerador_uva = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'], x=df_uva['Comprimento de Onda'])
        denominador_uva = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'] * 10**(-df_uva['Absorbancia'] * C), x=df_uva['Comprimento de Onda'])
        uva_pf = numerador_uva / denominador_uva

        st.success(f"**UVA-PF:** {uva_pf:.2f}")
        if 'spf_iso' in st.session_state:
            st.info(f"**UVA-PF/SPF:** {uva_pf / st.session_state.spf_iso:.2f} (Requisito ISO: ≥ 0.33)")

# =====================================================
# ABA 5: UVA1 & Ultra-Longo UVA (340–400 nm e 370–400 nm)
# =====================================================
with tab5:
    st.subheader("🔮 UVA1 (340–400 nm) & Ultra-Longo UVA (370–400 nm)")
    st.markdown("""
    **Baseado no estudo de HPC Today (2024):**
    - **UVA1-PF**: Fator de proteção na faixa 340–400 nm.
    - **Absorbância Média (370–400 nm)**: Valores ≥ 0.8 indicam boa proteção.
    """)

    if st.session_state.df_iso is not None:
        df = st.session_state.df_iso.copy()
        C = st.session_state.C_ajustado

        # (A) UVA1-PF (340–400 nm)
        df_uva1 = df[(df['Comprimento de Onda'] >= 340) & (df['Comprimento de Onda'] <= 400)]
        numer_uva1 = np.trapz(df_uva1['E(λ)'] * df_uva1['I(λ)'], x=df_uva1['Comprimento de Onda'])
        denom_uva1 = np.trapz(df_uva1['E(λ)'] * df_uva1['I(λ)'] * 10**(-df_uva1['Absorbancia'] * C), x=df_uva1['Comprimento de Onda'])
        uva1_pf = numer_uva1 / denom_uva1

        # (B) Absorbância Média (370–400 nm)
        df_ultra_long = df[df['Comprimento de Onda'] >= 370]
        absorbancia_media = df_ultra_long['Absorbancia'].mean()

        # Resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric("**UVA1-PF (340–400 nm)**", f"{uva1_pf:.2f}")
            if 'spf_iso' in st.session_state:
                st.metric("**UVA1-PF/SPF**", f"{uva1_pf / st.session_state.spf_iso:.2f}", 
                          "✅ ≥ 0.7 (Proteção UVA1 Superior)" if (uva1_pf / st.session_state.spf_iso) >= 0.7 else "⚠️ < 0.7")
        with col2:
            st.metric("**Absorbância Média (370–400 nm)**", f"{absorbancia_media:.2f}",
                      "✅ ≥ 0.8" if absorbancia_media >= 0.8 else "⚠️ < 0.8")

        # Gráfico
        fig, ax = plt.subplots()
        ax.plot(df['Comprimento de Onda'], df['Absorbancia'], label="Absorbância")
        ax.axvspan(340, 400, color='purple', alpha=0.1, label="UVA1 (340–400 nm)")
        ax.axvspan(370, 400, color='red', alpha=0.1, label="Ultra-Longo UVA (370–400 nm)")
        ax.set_xlabel("Comprimento de Onda (nm)")
        ax.set_ylabel("Absorbância")
        ax.legend()
        st.pyplot(fig)

# ===================== RODAPÉ =====================
st.markdown("---")
st.caption("""
**Referências:**  
- ISO 24443:2012 (*In vitro* UVA-PF)  
- Mansur et al. (1986) (*Fórmula rápida para SPF*)  
- HPC Today (2024) (*UVA1 e Ultra-Longo UVA*)  
""")
