import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Plataforma Científica de Fotoproteção", layout="wide")
st.title("☀️ Plataforma Científica de Fotoproteção")

# ===============================
# Funções matemáticas
# ===============================
def encontrar_coluna(df, nomes_possiveis):
    for nome in df.columns:
        if str(nome).strip().lower() in [n.lower() for n in nomes_possiveis]:
            return nome
    return None

def calcular_coeficiente_C(spf_in_vivo, spf_in_vitro):
    if spf_in_vitro == 0:
        return np.nan
    return spf_in_vivo / spf_in_vitro

def calcular_spf(wavelength, E, I, A):
    T = 10 ** (-A)
    num = np.sum(E * I)
    den = np.sum(E * I * T)
    return num / den

def calcular_uva_pf(wavelength, P, I, A):
    T = 10 ** (-A)
    num = np.sum(P * I)
    den = np.sum(P * I * T)
    return num / den

def calcular_uva1_pf(wavelength, P, I, A):
    mask = (wavelength >= 340)
    return calcular_uva_pf(wavelength[mask], P[mask], I[mask], A[mask])

def calcular_absorbancia_uva_longo(wavelength, A):
    mask = (wavelength >= 370)
    return np.mean(A[mask])

def calcular_razao_uva_spf(uva_pf, spf):
    return uva_pf / spf if spf != 0 else np.nan

def calcular_lambda_c(wavelength, A):
    T = 10 ** (-A)
    absorbance_area = np.cumsum(1 - T)
    total = absorbance_area[-1]
    idx = np.where(absorbance_area >= 0.5 * total)[0][0]
    return wavelength[idx]

# ===============================
# Upload de arquivo
# ===============================
uploaded_file = st.file_uploader("📂 Envie a planilha de dados (CSV ou XLSX)", type=["csv", "xlsx"])

# ===============================
# Aba de Orientações
# ===============================
tabs = st.tabs(["SPF", "UVA-PF", "UVA1-PF", "Absorbância UVA Longo", "Razão UVA/SPF", "λc", "Análise Completa", "Guia do Usuário"])
with tabs[7]:
    st.header("📘 Guia do Usuário")
    st.markdown("""
**Instruções de Upload:**
- Aceita arquivos CSV ou Excel.
- Colunas esperadas (nomes alternativos aceitos):
  - Comprimento de Onda (nm): wavelength, λ, nm, comprimento de onda
  - Absorbância (A): A, A0, absorbance, absorbância
  - E(λ): E, irradiância, spectral irradiance
  - I(λ): I, intensidade, solar
  - P(λ): P, ponderação, ppd
- Para dados brutos de transmitância: adicionar colunas `Transmitância_amostra` e `Transmitância_branco` e converter para absorbância.
- Faixas de comprimento de onda: UVB 290–320 nm, UVA 320–400 nm.
""")

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Pré-visualização dos dados:")
    st.dataframe(df.head())

    # Identificar colunas automaticamente
    col_lambda = encontrar_coluna(df, ["wavelength", "λ", "nm", "comprimento de onda"])
    col_E = encontrar_coluna(df, ["E", "spectral irradiance", "irradiancia"])
    col_I = encontrar_coluna(df, ["I", "solar", "intensidade"])
    col_A = encontrar_coluna(df, ["A", "A0", "absorbance", "absorbância"])
    col_P = encontrar_coluna(df, ["P", "ppd", "peso uva"])

    if not all([col_lambda, col_E, col_I, col_A, col_P]):
        st.error("❌ Não foi possível identificar todas as colunas necessárias. Verifique os nomes das colunas no arquivo.")
    else:
        wavelength = df[col_lambda].to_numpy()
        E = df[col_E].to_numpy()
        I = df[col_I].to_numpy()
        A = df[col_A].to_numpy()
        P = df[col_P].to_numpy()

        spf_in_vivo = st.number_input("Digite o valor de SPF in vivo (opcional):", min_value=0.0, value=0.0, step=0.1)

        # ===============================
        # Aba SPF
        # ===============================
        with tabs[0]:
            spf = calcular_spf(wavelength, E, I, A)
            C = calcular_coeficiente_C(spf_in_vivo, spf) if spf_in_vivo > 0 else np.nan
            st.subheader("📊 SPF (in vitro)")
            st.latex(r"SPF = \frac{\sum E(\lambda) I(\lambda)}{\sum E(\lambda) I(\lambda) T(\lambda)}")
            st.write(f"**SPF:** {spf:.2f}")
            if spf_in_vivo > 0:
                st.write(f"**Coeficiente C:** {C:.2f}")
            fig_spf = px.line(x=wavelength, y=10**(-A), labels={'x':'Comprimento de onda (nm)','y':'Transmitância T(λ)'}, title='Transmitância T(λ)')
            st.plotly_chart(fig_spf)

        # ===============================
        # Aba UVA-PF
        # ===============================
        with tabs[1]:
            uva_pf = calcular_uva_pf(wavelength, P, I, A)
            st.subheader("📊 UVA-PF")
            st.latex(r"UVA\text{-}PF = \frac{\sum P(\lambda) I(\lambda)}{\sum P(\lambda) I(\lambda) T(\lambda)}")
            st.write(f"**UVA-PF:** {uva_pf:.2f}")
            fig_uva = px.line(x=wavelength, y=P*I*10**(-A), labels={'x':'Comprimento de onda (nm)','y':'P(λ)·I(λ)·T(λ)'}, title='Contribuição ponderada UVA')
            st.plotly_chart(fig_uva)

        # ===============================
        # Aba UVA1-PF
        # ===============================
        with tabs[2]:
            uva1_pf = calcular_uva1_pf(wavelength, P, I, A)
            st.subheader("📊 UVA1-PF (340–400 nm)")
            st.write(f"**UVA1-PF:** {uva1_pf:.2f}")

        # ===============================
        # Aba Absorbância UVA Longo
        # ===============================
        with tabs[3]:
            A_uva_longo = calcular_absorbancia_uva_longo(wavelength, A)
            st.subheader("📊 Absorbância média UVA-Longo (370–400 nm)")
            st.write(f"**Absorbância média:** {A_uva_longo:.3f} {'✅' if A_uva_longo >= 0.8 else '⚠️'}")

        # ===============================
        # Aba Razão UVA/SPF
        # ===============================
        with tabs[4]:
            razao = calcular_razao_uva_spf(uva_pf, spf)
            st.subheader("📊 Razão UVA/SPF")
            st.write(f"**Razão UVA/SPF:** {razao:.2f} {'✅' if razao >= 1/3 else '⚠️'}")

        # ===============================
        # Aba λc
        # ===============================
        with tabs[5]:
            lambda_c = calcular_lambda_c(wavelength, A)
            st.subheader("📊 Comprimento de Onda Crítico (λc)")
            st.write(f"**λc:** {lambda_c:.2f} nm")

        # ===============================
        # Aba Análise Completa
        # ===============================
        with tabs[6]:
            st.subheader("📊 Análise Completa")
            st.write(f"**SPF:** {spf:.2f}")
            st.write(f"**UVA-PF:** {uva_pf:.2f}")
            st.write(f"**UVA1-PF:** {uva1_pf:.2f}")
            st.write(f"**Absorbância UVA-Longo:** {A_uva_longo:.3f}")
            st.write(f"**Razão UVA/SPF:** {razao:.2f}")
            st.write(f"**λc:** {lambda_c:.2f} nm")
           if spf_in_vivo >
                if spf_in_vivo > 0:
                st.write(f"**Coeficiente C:** {C:.2f}")

            # Botão para exportar resultados
            resultados = {
                "SPF": spf,
                "UVA-PF": uva_pf,
                "UVA1-PF": uva1_pf,
                "Absorbância UVA-Longo": A_uva_longo,
                "Razão UVA/SPF": razao,
                "λc (nm)": lambda_c,
                "Coeficiente C": C if spf_in_vivo > 0 else np.nan
            }

            if st.button("📥 Exportar Resultados para Excel"):
                output = BytesIO()
                df_result = pd.DataFrame(list(resultados.items()), columns=["Parâmetro", "Valor"])
                df_result.to_excel(output, index=False)
                st.download_button(label="Download Excel", data=output.getvalue(), file_name="resultados_fotoprotecao.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

