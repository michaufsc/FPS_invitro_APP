import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# ===================== CABEÇALHO COM LOGOS =====================
col1, col2 = st.columns([1, 0.5])
with col1:
    st.image("download.jpg", width=200)
with col2:
    st.image("download.png", width=200)

st.markdown("""
# 🌞 Ferramenta Avançada de Fotoproteção In Vitro
Este aplicativo permite calcular **SPF in vitro (ISO 24443 ajustado), SPF pelo método de Mansur e Comprimento de Onda Crítico (CWC)** usando a **mesma amostra espectrofotométrica**, oferecendo análises complementares.
""")

# ===================== ABAS =====================
tab1, tab2, tab3 = st.tabs(["ISO 24443 / Ajustado", "Mansur (1986)", "Comprimento de Onda Crítico"])

# ===================== Variáveis globais =====================
if "df_iso" not in st.session_state:
    st.session_state.df_iso = None
if "spf_iso" not in st.session_state:
    st.session_state.spf_iso = None

# =====================================================
# ABA 1 - ISO 24443 / Ajustado
# =====================================================
with tab1:
    st.subheader("📁 Cálculo do SPF in vitro - ISO 24443")
    with st.expander("ℹ️ Detalhes matemáticos e computacionais"):
        st.markdown("""
**Matemática:**  
O SPF in vitro é calculado como:

\[
SPF = \frac{\sum_{\lambda} E(\lambda) \cdot I(\lambda) \cdot \Delta\lambda}{\sum_{\lambda} E(\lambda) \cdot I(\lambda) \cdot T(\lambda) \cdot \Delta\lambda}
\]

- \(T(\lambda) = 10^{-A(\lambda)}\) é a transmitância.  
- \(E(\lambda)\) é o espectro de ação eritematosa.  
- \(I(\lambda)\) é a irradiância solar espectral.  
- \(\Delta\lambda\) é o passo entre comprimentos de onda.  

**Computação:**  
- `pandas` organiza os dados do usuário.  
- `numpy` realiza a soma ponderada (aproximação discreta da integral).  
- `matplotlib` plota a curva de transmitância.  
- `scipy.optimize` ajusta o coeficiente C minimizando a diferença entre SPF in vitro e SPF rotulado in vivo.

**Observação:**  
A mesma amostra usada aqui pode ser aplicada nas abas de Mansur e CWC.
""")
    uploaded_file = st.file_uploader("Carregue o arquivo Excel (.xlsx) com absorbância", type=["xlsx"], key="iso_upload")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            if 'Absorbancia' in df.columns:
                df['Absorbancia'] = df['Absorbancia'].astype(float)
                df['Transmitancia'] = 10 ** (-df['Absorbancia'])

                # SPF ISO
                d_lambda = df['Comprimento de Onda'][1] - df['Comprimento de Onda'][0]
                numerador = np.sum(df['E(λ)'] * df['I(λ)'] * d_lambda)
                denominador = np.sum(df['E(λ)'] * df['I(λ)'] * df['Transmitancia'] * d_lambda)

                if denominador != 0:
                    spf = numerador / denominador
                    st.success(f"🌞 SPF in vitro calculado: {spf:.2f}")
                    st.session_state.df_iso = df.copy()
                    st.session_state.spf_iso = spf
                else:
                    st.warning("⚠️ Denominador zero. Verifique os dados.")

                # Gráfico
                fig, ax = plt.subplots()
                ax.plot(df['Comprimento de Onda'], df['Transmitancia'], color='blue')
                ax.set_xlabel("Comprimento de Onda (nm)")
                ax.set_ylabel("Transmitância")
                ax.set_title("Transmitância vs Comprimento de Onda")
                ax.grid()
                st.pyplot(fig)

                # Ajuste SPF
                st.markdown("### 🔧 Ajuste do SPF in vitro")
                SPF_label = st.number_input("Insira o SPF rotulado in vivo", min_value=0.0, value=30.0)
                E = df['E(λ)'].to_numpy()
                I = df['I(λ)'].to_numpy()
                A0 = df['Absorbancia'].to_numpy()

                def spf_in_vitro_adj(C):
                    num = np.sum(E * I * d_lambda)
                    den = np.sum(E * I * 10**(-A0 * C) * d_lambda)
                    return num / den

                def error_function(C):
                    return abs(spf_in_vitro_adj(C) - SPF_label)

                result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                C_adjusted = result.x
                SPF_in_vitro_adj_final = spf_in_vitro_adj(C_adjusted)

                st.success(f"Coeficiente de ajuste C: {C_adjusted:.4f}")
                st.success(f"SPF in vitro ajustado: {SPF_in_vitro_adj_final:.2f}")
            else:
                st.error("❌ Coluna 'Absorbancia' não encontrada.")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

# =====================================================
# ABA 2 - Método de Mansur (1986)
# =====================================================
with tab2:
    st.subheader("📁 SPF pelo método de Mansur (1986)")
    with st.expander("ℹ️ Detalhes matemáticos e computacionais"):
        st.markdown("""
**Matemática:**  

\[
SPF = FC \times \sum_{\lambda=290}^{320} EE(\lambda) \cdot I(\lambda) \cdot A(\lambda)
\]

- \(A(\lambda)\) = absorbância da amostra.  
- \(EE(\lambda) \cdot I(\lambda)\) = fator espectral ponderado.  
- \(FC = 10\) = fator de correção.  

**Computação:**  
- `numpy` realiza soma discreta e produtos ponderados.  
- `pandas` filtra comprimentos de onda entre 290–320 nm automaticamente.  
- Rápido para avaliação de filtros químicos.  

**Observação:**  
Mesma amostra da Aba 1 pode ser reutilizada.
""")
    use_iso = st.checkbox("Usar dados da Aba 1 (ISO)")
    if use_iso and st.session_state.df_iso is not None:
        df_mansur = st.session_state.df_iso.copy()
    else:
        uploaded_mansur = st.file_uploader("Carregue arquivo para Mansur (.xlsx)", type=["xlsx"], key="mansur_upload")
        if uploaded_mansur:
            df_mansur = pd.read_excel(uploaded_mansur)
            df_mansur.columns = df_mansur.columns.str.strip()
        else:
            df_mansur = None

    if df_mansur is not None:
        df_mansur = df_mansur[(df_mansur['Comprimento de Onda'] >= 290) & (df_mansur['Comprimento de Onda'] <= 320)]
        ee_i_table = {290: 0.0150*0.134, 295:0.0817*0.134,300:0.2874*0.135,305:0.3278*0.136,310:0.1864*0.137,315:0.0839*0.138,320:0.0180*0.139}
        df_mansur['EE_I'] = df_mansur['Comprimento de Onda'].map(ee_i_table)
        df_mansur['Prod'] = df_mansur['EE_I'] * df_mansur['Absorbancia']
        FC = 10
        spf_mansur = FC * df_mansur['Prod'].sum()
        st.success(f"🌞 SPF (Mansur): {spf_mansur:.2f}")
        st.dataframe(df_mansur[['Comprimento de Onda','Absorbancia','EE_I','Prod']])

# =====================================================
# ABA 3 - Comprimento de Onda Crítico (CWC)
# =====================================================
with tab3:
    st.subheader("📁 Comprimento de Onda Crítico (CWC)")
    with st.expander("ℹ️ Detalhes matemáticos e computacionais"):
        st.markdown("""
**Matemática:**  

\[
CWC = \lambda \quad \text{onde} \quad \frac{\sum_{290}^{\lambda} A(\lambda) \Delta\lambda}{\sum_{290}^{400} A(\lambda) \Delta\lambda} = 0.9
\]

- Identifica o ponto em que 90% da área total da curva de absorbância é alcançada.  

**Computação:**  
- `numpy.cumsum` calcula área cumulativa.  
- `numpy.where` encontra índice correspondente.  
- `matplotlib` destaca CWC no gráfico.  

**Observação:**  
Pode usar a mesma amostra da Aba 1 ou novo arquivo.
""")
    use_iso_cwc = st.checkbox("Usar dados da Aba 1 (ISO)", key="use_iso_cwc")
    if use_iso_cwc and st.session_state.df_iso is not None:
        df_cwc = st.session_state.df_iso.copy()
    else:
        uploaded_cwc = st.file_uploader("Carregue arquivo para CWC (.xlsx)", type=["xlsx"], key="cwc_upload")
        if uploaded_cwc:
            df_cwc = pd.read_excel(uploaded_cwc)
            df_cwc.columns = df_cwc.columns.str.strip()
        else:
            df_cwc = None

    if df_cwc is not None:
        df_cwc = df_cwc[(df_cwc['Comprimento de Onda'] >= 290) & (df_cwc['Comprimento de Onda'] <= 400)]
        df_cwc['Absorbancia'] = df_cwc['Absorbancia'].astype(float)
        d_lambda = df_cwc['Comprimento de Onda'][1] - df_cwc['Comprimento de Onda'][0]
        area_total = np.sum(df_cwc['Absorbancia'] * d_lambda)
        area_cum = np.cumsum(df_cwc['Absorbancia'] * d_lambda)
        frac = area_cum / area_total
        idx_cwc = np.where(frac >= 0.9)[0][0]
        cwc_value = df_cwc['Comprimento de Onda'].iloc[idx_cwc]
        st.success(f"📏 Comprimento de Onda Crítico: {cwc_value:.1f} nm")
        fig, ax = plt.subplots()
        ax.plot(df_cwc['Comprimento de Onda'], df_cwc['Absorbancia'], label="Absorbância")
        ax.axvline(cwc_value, color='red', linestyle='--', label=f"CWC = {cwc_value:.1f} nm")
        ax.set_xlabel("Comprimento de Onda (nm)")
        ax.set_ylabel("Absorbância")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
