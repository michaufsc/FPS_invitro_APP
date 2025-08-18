import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Configuração da página
st.set_page_config(
    page_title="Cálculo de SPF e UVA-PF",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Calculadora Avançada de Protetor Solar")
st.markdown("""
**Calcule SPF, UVA-PF e λc** seguindo métodos internacionais:
- SPF: ISO 24444 (faixa 290-320 nm)
- UVA-PF: ISO 24442 (faixa 320-400 nm)
- λc (Comprimento de Onda Crítico): Método COLIPA
""")

# ======================================
# Funções principais
# ======================================
def encontrar_coluna(df, nomes_possiveis):
    """Identifica automaticamente colunas pelo nome"""
    for nome in df.columns:
        nome_limpo = str(nome).strip().lower()
        if nome_limpo in [n.lower() for n in nomes_possiveis]:
            return nome
    return None

def calcular_spf(wavelength, E, I, A):
    """Calcula SPF in vitro (290-320 nm)"""
    mask = (wavelength >= 290) & (wavelength <= 320)
    if not any(mask):
        return np.nan
    T = 10 ** (-A[mask])
    num = np.trapz(E[mask] * I[mask], wavelength[mask])
    den = np.trapz(E[mask] * I[mask] * T, wavelength[mask])
    return num / den if den != 0 else np.nan

def calcular_uva_pf(wavelength, P, I, A):
    """Calcula UVA-PF (320-400 nm)"""
    mask = (wavelength >= 320) & (wavelength <= 400)
    if not any(mask):
        return np.nan
    T = 10 ** (-A[mask])
    num = np.trapz(P[mask] * I[mask], wavelength[mask])
    den = np.trapz(P[mask] * I[mask] * T, wavelength[mask])
    return num / den if den != 0 else np.nan

def calcular_lambda_c(wavelength, A):
    """Calcula comprimento de onda crítico (λc)"""
    mask = (wavelength >= 290) & (wavelength <= 400)
    if not any(mask):
        return np.nan
    T = 10 ** (-A[mask])
    area = np.cumsum(1 - T[mask])
    total = area[-1]
    idx = np.where(area >= 0.9 * total)[0][0]  # 90% da área
    return wavelength[mask][idx]

def gerar_template():
    """Gera um template de dados para download"""
    wavelength = np.arange(290, 401, 5)
    return pd.DataFrame({
        "Wavelength (nm)": wavelength,
        "E (irradiancia)": np.linspace(0.02, 0.001, len(wavelength)),
        "I (intensidade solar)": np.linspace(1.0, 0.1, len(wavelength)),
        "A (absorbância)": np.linspace(0.1, 1.5, len(wavelength)),
        "P (peso UVA)": [0.5 if λ >= 320 else 0 for λ in wavelength]
    })

# ======================================
# Interface do usuário
# ======================================
with st.sidebar:
    st.header("⚙️ Configurações")
    st.download_button(
        label="📥 Baixar Template",
        data=gerar_template().to_csv(index=False).encode('utf-8'),
        file_name="template_protetor_solar.csv",
        mime="text/csv"
    )
    st.markdown("""
    **Formato esperado:**
    - Coluna com comprimento de onda (nm)
    - Coluna E: Irradiação espectral (W/m²/nm)
    - Coluna I: Intensidade solar relativa
    - Coluna A: Absorbância
    - Coluna P: Peso UVA (opcional)
    """)

# Upload de arquivo
uploaded_file = st.file_uploader(
    "📤 Carregue seu arquivo de dados (CSV ou Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    try:
        # Leitura do arquivo
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Verificação básica de dados
        if df.empty:
            st.error("⚠️ O arquivo está vazio!")
            st.stop()
            
        if len(df) < 10:
            st.warning("Poucos dados para cálculos precisos (recomendado ≥30 pontos)")

        # Identificação automática de colunas
        col_lambda = encontrar_coluna(df, ["wavelength", "nm", "comprimento de onda"])
        col_E = encontrar_coluna(df, ["E", "irradiancia", "spectral irradiance"])
        col_I = encontrar_coluna(df, ["I", "intensidade", "solar intensity"])
        col_A = encontrar_coluna(df, ["A", "absorbance", "absorbância"])
        col_P = encontrar_coluna(df, ["P", "ppd", "peso uva"])

        if not all([col_lambda, col_E, col_I, col_A]):
            st.error("❌ Colunas essenciais não encontradas!")
            st.write("Colunas detectadas:", df.columns.tolist())
            st.stop()

        # Processamento dos dados
        wavelength = df[col_lambda].astype(float).to_numpy()
        E = df[col_E].astype(float).to_numpy()
        I = df[col_I].astype(float).to_numpy()
        A = df[col_A].astype(float).to_numpy()
        P = df[col_P].astype(float).to_numpy() if col_P else np.ones_like(wavelength)

        # Cálculos
        with st.spinner("Calculando parâmetros..."):
            spf = calcular_spf(wavelength, E, I, A)
            uva_pf = calcular_uva_pf(wavelength, P, I, A)
            lambda_c = calcular_lambda_c(wavelength, A)

        # Exibição de resultados
        st.success("✅ Cálculos concluídos!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SPF in vitro", f"{spf:.1f}" if not np.isnan(spf) else "N/A")
        with col2:
            st.metric("UVA-PF", f"{uva_pf:.1f}" if not np.isnan(uva_pf) else "N/A")
        with col3:
            st.metric("λc (nm)", f"{lambda_c:.1f}" if not np.isnan(lambda_c) else "N/A")

        # Gráficos
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Gráfico de absorbância
        ax1.plot(wavelength, A, label="Absorbância", color='royalblue', linewidth=2)
        ax1.axvspan(290, 320, color='red', alpha=0.1, label='UVB (290-320 nm)')
        ax1.axvspan(320, 400, color='purple', alpha=0.1, label='UVA (320-400 nm)')
        ax1.axvline(lambda_c, color='black', linestyle='--', label=f'λc = {lambda_c:.1f} nm')
        ax1.set_title("Perfil de Absorbância")
        ax1.set_xlabel("Comprimento de Onda (nm)")
        ax1.set_ylabel("Absorbância")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico de transmitância
        ax2.plot(wavelength, 10**(-A), label="Transmitância", color='green', linewidth=2)
        ax2.set_title("Perfil de Transmitância")
        ax2.set_xlabel("Comprimento de Onda (nm)")
        ax2.set_ylabel("Transmitância")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Exportação de resultados
        result_df = pd.DataFrame({
            "Parâmetro": ["SPF in vitro", "UVA-PF", "λc (nm)"],
            "Valor": [spf, uva_pf, lambda_c]
        })

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='Resultados', index=False)
            df.to_excel(writer, sheet_name='Dados Originais', index=False)
        
        st.download_button(
            label="📊 Exportar Resultados (Excel)",
            data=output.getvalue(),
            file_name="resultados_fotoprotecao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Erro durante o processamento: {str(e)}")
        st.write("Verifique o formato dos dados e tente novamente.")

else:
    st.info("ℹ️ Carregue um arquivo para começar a análise")
    st.image("https://via.placeholder.com/800x400?text=Exemplo+de+Dados+UV", 
             caption="Exemplo de espectro UV típico para protetores solares")
