import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Título do aplicativo
st.title("Cálculo do SPF In Vitro e Ajuste do SPF")

# Função para carregar os dados do arquivo Excel
uploaded_file = st.file_uploader("📁 Faça o upload do arquivo Excel contendo os dados...", type=["xlsx"])

# Verificar se o arquivo foi carregado
if uploaded_file:
    try:
        # Carregar os dados
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()  # Remove espaços extras nos nomes das colunas
        st.write("Colunas disponíveis no arquivo:", df.columns)

        # Ajustar os nomes das colunas conforme necessário
        comprimento_onda_coluna = 'Comprimento de Onda'  # Ajuste se necessário
        E_coluna = 'E(λ)'  # Ajuste se necessário
        I_coluna = 'I(λ)'  # Ajuste se necessário
        A0_coluna = 'Absorbancia'  # Ajuste se necessário

        # Verificar e extrair as colunas necessárias
        comprimento_onda = df[comprimento_onda_coluna].to_numpy()
        E = df[E_coluna].to_numpy()
        I = df[I_coluna].to_numpy()
        A0 = df[A0_coluna].to_numpy()

        # Solicitar comprimentos de onda de início e fim para análise
        inicio_onda = st.number_input("Insira o comprimento de onda inicial (nm):", min_value=float(comprimento_onda.min()), max_value=float(comprimento_onda.max()))
        fim_onda = st.number_input("Insira o comprimento de onda final (nm):", min_value=float(comprimento_onda.min()), max_value=float(comprimento_onda.max()))

        # Filtrar os dados para o intervalo de comprimento de onda desejado
        filtro = (comprimento_onda >= inicio_onda) & (comprimento_onda <= fim_onda)
        E = E[filtro]
        I = I[filtro]
        A0 = A0[filtro]

        # Exibir os dados filtrados
        st.subheader("📊 Dados filtrados para o intervalo de comprimento de onda")
        st.dataframe(pd.DataFrame({
            comprimento_onda_coluna: comprimento_onda[filtro],
            E_coluna: E,
            I_coluna: I,
            A0_coluna: A0
        }))

        # Calcular a Transmitância
        df['Transmitancia'] = 10 ** (-df[A0_coluna])
        st.subheader("📊 Tabela com Transmitância")
        st.dataframe(df)

        st.subheader("📈 Gráfico: Transmitância vs Comprimento de Onda")
        fig, ax = plt.subplots()
        ax.plot(df[comprimento_onda_coluna], df['Transmitancia'], color='blue')
        ax.set_xlabel("Comprimento de Onda (nm)")
        ax.set_ylabel("Transmitância")
        ax.set_title("Transmitância vs Comprimento de Onda")
        ax.grid()
        st.pyplot(fig)

        # Calcular o SPF In Vitro
        if all(col in df.columns for col in ['E(λ)', 'I(λ)', 'Transmitancia']):
            d_lambda = df[comprimento_onda_coluna][1] - df[comprimento_onda_coluna][0]
            numerador = np.sum(df['E(λ)'] * df['I(λ)'] * d_lambda)
            denominador = np.sum(df['E(λ)'] * df['I(λ)'] * df['Transmitancia'] * d_lambda)

            if denominador != 0:
                spf = numerador / denominador
                st.subheader(f"🌞 SPF in vitro calculado: {spf:.2f}")
            else:
                st.warning("⚠️ O denominador é zero. Verifique os dados.")
        else:
            st.error("❌ A planilha deve conter: 'E(λ)', 'I(λ)', 'Transmitancia'.")

        # Agora, calcular o SPF Ajustado In Vitro
        st.subheader("🌞 Calcular o SPF Ajustado In Vitro")

        # Informar que o SPF in vitro mostrado é o calculado anteriormente
        st.info("📌 O valor do SPF in vitro que será utilizado no ajuste é o valor calculado acima.")

        # Solicitar valores de SPF in vivo e SPF in vitro
        SPF_label = st.number_input("Insira o valor do SPF in vivo (SPF_label):", min_value=0.0)
        SPF_in_vitro = spf  # Utiliza o valor do SPF calculado anteriormente

        # Função para calcular o SPF ajustado in vitro
        def spf_in_vitro_adj(C):
            d_lambda = 1  # Passo de comprimento de onda (1 nm)
            numerador = np.sum(E * I * d_lambda)
            denominador = np.sum(E * I * 10**(-A0 * C) * d_lambda)
            return numerador / denominador

        # Função de erro para otimização
        def error_function(C):
            return abs(spf_in_vitro_adj(C) - SPF_label)

        # Otimização para encontrar o valor de C
        result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')

        # Resultado
        C_adjusted = result.x
        SPF_in_vitro_adj_final = spf_in_vitro_adj(C_adjusted)

        # Exibir resultados
        st.subheader(f"🌞 Coeficiente de ajuste C: {C_adjusted:.4f}")
        st.subheader(f"🌞 SPF in vitro ajustado: {SPF_in_vitro_adj_final:.4f}")
        st.subheader(f"🌞 SPF desejado (label): {SPF_label}")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

