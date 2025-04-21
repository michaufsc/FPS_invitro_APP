import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mensagem inicial
st.title("🌞Cálculo do SPF In Vitro a partir de Dados Espectrais🌞")

st.markdown("""
Este aplicativo permite que o usuário envie uma planilha Excel contendo dados espectrais (Absorbância, Comprimento de Onda, E(λ) e I(λ)) para calcular a Transmitância e o SPF in vitro automaticamente.

### Como usar
1. Suba um arquivo Excel (.xlsx) com as colunas:
   - Absorbancia
   - Comprimento de Onda
   - E(λ)
   - I(λ)

2. O app calcula a transmitância com base na Lei de Lambert-Beer e exibe o valor do SPF in vitro.

3. Gráficos e tabela são exibidos para análise visual.

Feito com ❤️ usando Streamlit
""")

# Carregar o arquivo Excel
uploaded_file = st.file_uploader("📁 Faça o upload do arquivo Excel (.xlsx)", type=["xlsx"])

# Verificar se o arquivo foi carregado
if uploaded_file:
    try:
        # Ler o arquivo Excel
        df = pd.read_excel(uploaded_file)
        # Limpar espaços extras nos nomes das colunas
        df.columns = df.columns.str.strip()

        # Verificar se a coluna 'Absorbancia' está presente
        if 'Absorbancia' in df.columns:
            # Calcular a Transmitância
            df['Absorbancia'] = df['Absorbancia'].astype(float)
            df['Transmitancia'] = 10 ** (-df['Absorbancia'])
            st.success("✅ Transmitância calculada com sucesso!")

            # Verificar se as colunas necessárias estão presentes
            if all(col in df.columns for col in ['Comprimento de Onda', 'Transmitancia', 'E(λ)', 'I(λ)']):
                # Calcular o valor de dλ
                d_lambda = df['Comprimento de Onda'][1] - df['Comprimento de Onda'][0]

                # Soma de Riemann para calcular o SPF
                numerador = np.sum(df['E(λ)'] * df['I(λ)'] * d_lambda)
                denominador = np.sum(df['E(λ)'] * df['I(λ)'] * df['Transmitancia'] * d_lambda)

                # Calcular o SPF se o denominador não for zero
                if denominador != 0:
                    spf = numerador / denominador
                    st.subheader(f"🌞 SPF in vitro calculado: {spf:.2f}")
                else:
                    st.warning("⚠️ O denominador é zero. Verifique os dados.")

            else:
                st.error("❌ A planilha deve conter as colunas: 'Comprimento de Onda', 'E(λ)', 'I(λ)', e 'Transmitancia'.")

            # Exibir a tabela com a transmitância calculada
            st.subheader("📊 Tabela com Transmitância")
            st.dataframe(df)

            # Exibir o gráfico de Transmitância vs Comprimento de Onda
            st.subheader("📈 Gráfico: Transmitância vs Comprimento de Onda")
            fig, ax = plt.subplots()
            ax.plot(df['Comprimento de Onda'], df['Transmitancia'], color='blue')
            ax.set_xlabel("Comprimento de Onda (nm)")
            ax.set_ylabel("Transmitância")
            ax.set_title("Transmitância vs Comprimento de Onda")
            ax.grid()
            st.pyplot(fig)

        else:
            st.error("❌ A coluna 'Absorbancia' não foi encontrada no arquivo.")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

