import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# LOGOS NO TOPO
col1, col2 = st.columns([1, 0.5])
with col1:
    st.image("download.jpg", width=200)
with col2:
    st.image("download.png", width=200)

# TÍTULO PRINCIPAL
st.markdown("""
# 🌞 Cálculo do SPF in vitro  
### 🌞🌞Cálculo do SPF in vitro ajustado (SPF in vitro ajus)  
### 🌞🌞🌞Determinação do coeficiente de ajuste ‘C’
""")

# ETAPA 1: Upload do arquivo
st.markdown("### 📁 Etapa 1: Envio da planilha com dados espectrais")
uploaded_file = st.file_uploader("Faça o upload do arquivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()

        # ETAPA 2: Cálculo da Transmitância
        st.markdown("### 🔬 Etapa 2: Cálculo da Transmitância")
        if 'Absorbancia' in df.columns:
            df['Absorbancia'] = df['Absorbancia'].astype(float)
            df['Transmitancia'] = 10 ** (-df['Absorbancia'])
            st.success("✅ Transmitância calculada com sucesso!")

            # ETAPA 3: Cálculo do SPF in vitro
            st.markdown("### 🧮 Etapa 3: Cálculo do SPF in vitro")
            if all(col in df.columns for col in ['Comprimento de Onda', 'Transmitancia', 'E(λ)', 'I(λ)']):
                d_lambda = df['Comprimento de Onda'][1] - df['Comprimento de Onda'][0]
                numerador = np.sum(df['E(λ)'] * df['I(λ)'] * d_lambda)
                denominador = np.sum(df['E(λ)'] * df['I(λ)'] * df['Transmitancia'] * d_lambda)

                if denominador != 0:
                    spf = numerador / denominador
                    st.subheader(f"🌞 SPF in vitro calculado: {spf:.2f}")
                else:
                    st.warning("⚠️ O denominador é zero. Verifique os dados.")
            else:
                st.error("❌ A planilha deve conter: 'Comprimento de Onda', 'E(λ)', 'I(λ)'.")
            
            # ETAPA 4: Visualização
            st.markdown("### 📊 Etapa 4: Visualização dos dados")
            st.dataframe(df)
            fig, ax = plt.subplots()
            ax.plot(df['Comprimento de Onda'], df['Transmitancia'], color='blue')
            ax.set_xlabel("Comprimento de Onda (nm)")
            ax.set_ylabel("Transmitância")
            ax.set_title("Transmitância vs Comprimento de Onda")
            ax.grid()
            st.pyplot(fig)

            # ETAPA 5: Ajuste do SPF in vitro
            st.markdown("### 🔧 Etapa 5: Ajuste do SPF in vitro (SPF in vitro adjus)")
            st.info("O valor de SPF in vitro usado aqui é o que foi calculado na etapa anterior.")

            SPF_label = st.number_input("Insira o valor do SPF in vivo (SPF_label)", min_value=0.0, value=30.0)
            SPF_in_vitro = spf

            comprimento_onda = df['Comprimento de Onda'].to_numpy()
            E = df['E(λ)'].to_numpy()
            I = df['I(λ)'].to_numpy()
            A0 = df['Absorbancia'].to_numpy()

            def spf_in_vitro_adj(C):
                numerador = np.sum(E * I * d_lambda)
                denominador = np.sum(E * I * 10**(-A0 * C) * d_lambda)
                return numerador / denominador

            def error_function(C):
                return abs(spf_in_vitro_adj(C) - SPF_label)

            result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
            C_adjusted = result.x
            SPF_in_vitro_adj_final = spf_in_vitro_adj(C_adjusted)

            st.success(f"🔢 Coeficiente de ajuste C: {C_adjusted:.4f}")
            st.success(f"✅ SPF in vitro ajustado: {SPF_in_vitro_adj_final:.2f}")
            st.info(f"🎯 SPF rotulado in vivo (label): {SPF_label}")

        else:
            st.error("❌ A coluna 'Absorbancia' não foi encontrada.")
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

# ETAPA 6: Cálculo do UVA-PF
st.markdown("---")
st.markdown("### ☀️ Etapa 6: Cálculo do UVA-PF com coeficiente C (insira o valor do coeficiente C calculado na etapa 5)")
st.warning("⚠️ Atenção: Esta etapa utiliza **outro arquivo onde os valores P e I das colunas são tabelados conforme a ISO/FDIS 24443:2011(E)**, Os valores são diferentes do cálculo de FPS inicial. Faça novo upload com as colunas 'P', 'I' e 'A_e'.")

uva_file = st.file_uploader("📁 Faça o upload do arquivo com os dados para o UVA-PF (.csv)", type=["csv"], key="uva_pf_upload")

if uva_file:
    try:
        df_uva = pd.read_csv(uva_file, delimiter=';')
        df_uva.columns = df_uva.columns.str.strip()

        df_uva['P'] = pd.to_numeric(df_uva['P'].astype(str).str.replace('\n', '').str.replace(',', '.'), errors='coerce')
        df_uva['I'] = pd.to_numeric(df_uva['I'].astype(str).str.replace('\n', '').str.replace(',', '.'), errors='coerce')
        df_uva['A_e'] = pd.to_numeric(df_uva['A_e'].astype(str).str.replace('\n', '').str.replace(',', '.'), errors='coerce')
        df_uva = df_uva.dropna(subset=['P', 'I', 'A_e'])

        C_manual = st.number_input("Insira o valor do coeficiente C para cálculo do UVA-PF", min_value=0.0, max_value=5.0, value=0.8, step=0.01)

        d_lambda = 1  # Passo fixo de 1 nm
        P = df_uva['P'].to_numpy()
        I = df_uva['I'].to_numpy()
        A_e = df_uva['A_e'].to_numpy()

        numerador = np.sum(P * I * d_lambda)
        denominador = np.sum(P * I * 10**(-A_e * C_manual) * d_lambda)

        if denominador != 0:
            uva_pf = numerador / denominador
            st.success(f"🌿 UVA-PF calculado: {uva_pf:.2f}")
        else:
            st.warning("⚠️ O denominador do cálculo do UVA-PF é zero. Verifique os dados.")
    except Exception as e:
        st.error(f"Erro ao processar o arquivo do UVA-PF: {e}")

         
             
