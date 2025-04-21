import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# TÍTULO PRINCIPAL
st.markdown("""
# 🌞 Cálculo do SPF in vitro  
### Cálculo do SPF in vitro ajustado (SPF in vitro ajus)  
### Determinação do coeficiente de ajuste ‘C’
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

            # Exibir tabela e gráfico
            st.markdown("### 📊 Etapa 4: Visualização dos dados")
            st.dataframe(df)

            fig, ax = plt.subplots()
            ax.plot(df['Comprimento de Onda'], df['Transmitancia'], color='blue')
            ax.set_xlabel("Comprimento de Onda (nm)")
            ax.set_ylabel("Transmitância")
            ax.set_title("Transmitância vs Comprimento de Onda")
            ax.grid()
            st.pyplot(fig)

            # ETAPA 5: Ajuste do SPF in vitro (SPF in vitro ajus)
            st.markdown("### 🔧 Etapa 5: Ajuste do SPF in vitro (SPF in vitro ajus)")
            st.info("O valor de SPF in vitro usado aqui é o que foi calculado na etapa anterior.")

            SPF_label = st.number_input("Insira o valor do SPF in vivo (SPF_label)", min_value=0.0, value=30.0)
            SPF_in_vitro = spf  # Valor já calculado

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
