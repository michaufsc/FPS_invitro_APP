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
# 🌞 Cálculo de Fotoproteção In Vitro
Ferramenta para cálculo de SPF, ajuste in vitro, método de Mansur e Comprimento de Onda Crítico.
""")

# ===================== ABAS =====================
tab1, tab2, tab3 = st.tabs(["ISO 24443 / Ajustado", "Mansur (1986)", "Comprimento de Onda Crítico"])

# Variáveis globais
if "df_iso" not in st.session_state:
    st.session_state.df_iso = None
if "spf_iso" not in st.session_state:
    st.session_state.spf_iso = None

# =====================================================
# ABA 1 - ISO 24443 / Ajustado
# =====================================================
with tab1:
    st.subheader("📁 Cálculo do SPF in vitro - ISO 24443")
    with st.expander("ℹ️ Sobre esta metodologia"):
        st.markdown("""
        **Objetivo:** Determinar o SPF in vitro de protetores solares e ajustar os resultados para melhor
        correspondência com o SPF in vivo usando o coeficiente C.

        **Passos para o usuário:**
        1. Medir a absorbância da amostra aplicada sobre substrato adequado (ex.: placa de PMMA) em comprimentos de onda de 290 a 400 nm.
        2. Salvar os dados em Excel com colunas:
           - `Comprimento de Onda` (nm)
           - `Absorbancia`
           - `E(λ)` (espectro de ação eritematosa)
           - `I(λ)` (irradiância solar)
        3. Fazer upload do arquivo.
        4. Conferir tabela e gráfico de transmitância gerados.
        5. Inserir o SPF rotulado in vivo para cálculo do coeficiente C.
        6. Observar o SPF in vitro ajustado e o gráfico da curva.

        **Referências:**
        - ISO 24443:2011(E) – “Sun protection test methods in vitro”.
        - Mansur, J.S., et al., 1986. *An. Bras. Dermatol.*, 61(3), pp. 121–124.
        """)

    uploaded_file = st.file_uploader("Carregue o arquivo Excel (.xlsx)", type=["xlsx"], key="iso_upload")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()

            if 'Absorbancia' in df.columns:
                df['Absorbancia'] = df['Absorbancia'].astype(float)
                df['Transmitancia'] = 10 ** (-df['Absorbancia'])

                # Cálculo SPF ISO
                d_lambda = df['Comprimento de Onda'][1] - df['Comprimento de Onda'][0]
                numerador = np.sum(df['E(λ)'] * df['I(λ)'] * d_lambda)
                denominador = np.sum(df['E(λ)'] * df['I(λ)'] * df['Transmitancia'] * d_lambda)

                if denominador != 0:
                    spf = numerador / denominador
                    st.success(f"🌞 SPF in vitro calculado: {spf:.2f}")

                    # Guardar dados para uso nas outras abas
                    st.session_state.df_iso = df.copy()
                    st.session_state.spf_iso = spf
                else:
                    st.warning("⚠️ O denominador é zero. Verifique os dados.")

                # Gráfico
                fig, ax = plt.subplots()
                ax.plot(df['Comprimento de Onda'], df['Transmitancia'], color='blue')
                ax.set_xlabel("Comprimento de Onda (nm)")
                ax.set_ylabel("Transmitância")
                ax.set_title("Transmitância vs Comprimento de Onda")
                ax.grid()
                st.pyplot(fig)

                # Ajuste de SPF
                st.markdown("### 🔧 Ajuste do SPF in vitro")
                SPF_label = st.number_input("Insira o SPF in vivo rotulado", min_value=0.0, value=30.0)
                comprimento_onda = df['Comprimento de Onda'].to_numpy()
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
                st.error("❌ A coluna 'Absorbancia' não foi encontrada.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

# =====================================================
# ABA 2 - Método de Mansur (1986)
# =====================================================
with tab2:
    st.subheader("📁 Cálculo do SPF - Método de Mansur (1986)")
    with st.expander("ℹ️ Sobre esta metodologia"):
        st.markdown("""
        **Objetivo:** Determinar o SPF in vitro para formulações com filtros solares químicos de forma rápida.

        **Passos para o usuário:**
        1. Preparar a amostra em álcool (0,2 μL/mL) ou éter para filtros oleosos.
        2. Medir absorbância entre 290 e 320 nm em intervalos de 5 nm.
        3. Salvar os dados em Excel com colunas:
           - `Comprimento de Onda` (nm)
           - `Absorbancia`
        4. Fazer upload ou usar dados da Aba 1.
        5. O aplicativo calcula SPF automaticamente com FC = 10.

        **Referências:**
        - Mansur, J.S., et al., 1986. *An. Bras. Dermatol.*, 61(3), pp. 121–124.
        """)

    use_iso = st.checkbox("Usar dados carregados na Aba 1 (ISO)")
    if use_iso and st.session_state.df_iso is not None:
        df_mansur = st.session_state.df_iso.copy()
    else:
        uploaded_mansur = st.file_uploader("Carregue o arquivo para Mansur (.xlsx)", type=["xlsx"], key="mansur_upload")
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
    st.subheader("📁 Cálculo do Comprimento de Onda Crítico (CWC)")
    with st.expander("ℹ️ Sobre esta metodologia"):
        st.markdown("""
        **Objetivo:** Determinar o comprimento de onda em que 90% da proteção UV é fornecida.

        **Passos para o usuário:**
        1. Usar dados de absorbância da Aba ISO ou fazer upload de arquivo separado.
        2. Garantir que a coluna `Absorbancia` esteja presente.
        3. O aplicativo calcula a área cumulativa e identifica o CWC.
        4. Conferir o gráfico com a linha vertical indicando o CWC.

        **Referências:**
        - Diffey, B.L., Robson, J., 1989. *Int J Cosmet Sci*, 11, pp. 283–292.
        - ISO 24443:2011(E) – “Sun protection test methods in vitro”.
        """)

    use_iso_cwc = st.checkbox("Usar dados carregados na Aba 1 (ISO)", key="use_iso_cwc")
    if use_iso_cwc and st.session_state.df_iso is not None:
        df_cwc = st.session_state.df_iso.copy()
    else:
        uploaded_cwc = st.file_uploader("Carregue o arquivo para CWC (.xlsx)", type=["xlsx"], key="cwc_upload")
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


         
             
