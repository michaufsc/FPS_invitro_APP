import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("🧴 Cálculo do SPF In Vitro a partir de Dados Espectrais")

uploaded_file = st.file_uploader("📁 Faça o upload do arquivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()  # Remove espaços nos nomes das colunas

        if 'Absorbancia' in df.columns:
            df['Absorbancia'] = df['Absorbancia'].astype(float)
            df['Transmitancia'] = 10 ** (-df['Absorbancia'])

            st.success("✅ Transmitância calculada com sucesso!")

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

            st.subheader("📊 Tabela com Transmitância")
            st.dataframe(df)

            st.subheader("📈 Gráfico: Transmitância vs Comprimento de Onda")
            fig, ax = plt.subplots()
            ax.plot(df['Comprimento de Onda'], df['Transmitancia'], color='blue')
            ax.set_xlabel("Comprimento de Onda (nm)")
            ax.set_ylabel("Transmitância")
            ax.set_title("Transmitância vs Comprimento de Onda")
            ax.grid()
            st.pyplot(fig)

        else:
            st.error("❌ A coluna 'Absorbancia' não foi encontrada.")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
