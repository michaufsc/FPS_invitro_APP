import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from io import BytesIO
from fpdf import FPDF  # Para geração de relatórios PDF

# =============================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================
st.set_page_config(
    page_title="Plataforma Científica de Fotoproteção",
    page_icon="🔬",
    layout="wide"
)

# =============================================
# CONSTANTES CIENTÍFICAS
# =============================================
COEF_MANSUR = {
    290: 0.00201, 295: 0.01095, 300: 0.03880,
    305: 0.04458, 310: 0.02554, 315: 0.01158, 320: 0.00250
}

# =============================================
# FUNÇÕES CIENTÍFICAS
# =============================================
def corrigir_absorbancia(transmitancia_amostra, transmitancia_branco):
    """Converte transmitância bruta em absorbância"""
    return -np.log10(np.array(transmitancia_amostra) / np.array(transmitancia_branco))

def calcular_spf(df, C=1.0, lambda_min=290, lambda_max=320):
    """Calcula SPF com incerteza (ISO 24443)"""
    df_faixa = df[(df['Comprimento de Onda (nm)'] >= lambda_min) & 
                 (df['Comprimento de Onda (nm)'] <= lambda_max)].copy()
    
    num = np.trapz(df_faixa['E(λ)'] * df_faixa['I(λ)'], df_faixa['Comprimento de Onda (nm)'])
    den = np.trapz(df_faixa['E(λ)'] * df_faixa['I(λ)'] * 10**(-df_faixa['Absorbância'] * C), 
                  df_faixa['Comprimento de Onda (nm)'])
    spf = num / den
    
    # Simulação de incerteza (Monte Carlo)
    spfs = []
    for _ in range(100):
        A_perturbada = df_faixa['Absorbância'] * np.random.normal(1, st.session_state.ruido/100)
        den_perturbado = np.trapz(df_faixa['E(λ)'] * df_faixa['I(λ)'] * 10**(-A_perturbada * C), 
                                 df_faixa['Comprimento de Onda (nm)'])
        spfs.append(num / den_perturbado)
    
    return spf, np.std(spfs)

def calcular_uva_pf(df, C=1.0):
    """Calcula UVA-PF (ISO 24443)"""
    df_uva = df[(df['Comprimento de Onda (nm)'] >= 320)].copy()
    num = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'], df_uva['Comprimento de Onda (nm)'])
    den = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'] * 10**(-df_uva['Absorbância'] * C), 
                  df_uva['Comprimento de Onda (nm)'])
    return num / den

def calcular_uva1_pf(df, C=1.0):
    """Calcula UVA1-PF (340-400 nm) - HPC Today 2024"""
    df_uva1 = df[(df['Comprimento de Onda (nm)'] >= 340)].copy()
    return calcular_uva_pf(df_uva1, C)

def calcular_cwc(df):
    """Calcula Comprimento de Onda Crítico (ISO 24443)"""
    df_faixa = df[(df['Comprimento de Onda (nm)'] >= 290) & 
                 (df['Comprimento de Onda (nm)'] <= 400)].copy()
    
    area_total = np.trapz(df_faixa['Absorbância'], df_faixa['Comprimento de Onda (nm)'])
    area_cum = np.cumsum(df_faixa['Absorbância'] * np.gradient(df_faixa['Comprimento de Onda (nm)']))
    
    idx = np.where(area_cum >= 0.9 * area_total)[0][0]
    return df_faixa['Comprimento de Onda (nm)'].iloc[idx]

def calcular_mansur(df):
    """Calcula SPF pelo método de Mansur (1986)"""
    df['Contribuição'] = df['Comprimento de Onda (nm)'].map(COEF_MANSUR) * df['Absorbância']
    return 10 * df['Contribuição'].sum()

def gerar_relatorio(resultados):
    """Gera relatório em PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Relatório de Fotoproteção", ln=1, align='C')
    pdf.cell(200, 10, txt="="*50, ln=1, align='C')
    
    for metodo, valor in resultados.items():
        pdf.cell(200, 10, txt=f"{metodo}: {valor}", ln=1)
    
    return pdf.output(dest='S').encode('latin1')

# =============================================
# INTERFACE DO USUÁRIO
# =============================================
st.title("🔬 Plataforma Científica de Fotoproteção")
st.markdown("""
**Métodos disponíveis:**  
✔ SPF (ISO 24443)  
✔ UVA-PF  
✔ UVA1-PF (340-400 nm)  
✔ CWC  
✔ Método Rápido (Mansur)  
✔ Análise Completa  
""")

# Configurações na sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    st.session_state.ruido = st.slider("Nível de ruído para incerteza (%):", 0.1, 5.0, 1.0)
    
    tipo_entrada = st.radio(
        "Tipo de dados de entrada:",
        ["Absorbância pronta", "Transmitância bruta (com branco)"]
    )

# Upload de dados
uploaded_file = st.file_uploader("📤 Upload do arquivo Excel", type=["xlsx"])

# Processamento dos dados
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    if tipo_entrada == "Transmitância bruta (com branco)":
        try:
            df['Absorbância'] = corrigir_absorbancia(
                df['Transmitância_amostra'], 
                df['Transmitância_branco']
            )
            st.success("✅ Absorbância calculada a partir dos dados brutos!")
        except KeyError:
            st.error("Erro: Arquivo não contém as colunas 'Transmitância_amostra' e 'Transmitância_branco'")

# Seleção de métodos
metodo = st.selectbox(
    "Selecione o método de análise:",
    [
        "SPF (ISO 24443)", 
        "UVA-PF", 
        "UVA1-PF (340-400 nm)", 
        "CWC", 
        "Método Rápido (Mansur)", 
        "Análise Completa"
    ]
)

# Cálculos e resultados
if uploaded_file and st.button("Calcular"):
    resultados = {}
    C = 1.0
    
    if metodo in ["SPF (ISO 24443)", "Análise Completa"]:
        spf_in_vivo = st.number_input("SPF in vivo conhecido (para ajuste):", value=30.0)
        C = opt.minimize_scalar(lambda C: abs(calcular_spf(df, C)[0] - spf_in_vivo)).x
        spf, incerteza = calcular_spf(df, C)
        resultados["SPF (ISO 24443)"] = f"{spf:.2f} ± {incerteza:.2f} (C={C:.3f})"
    
    if metodo in ["UVA-PF", "Análise Completa"]:
        uva_pf = calcular_uva_pf(df, C)
        resultados["UVA-PF"] = f"{uva_pf:.2f}"
    
    if metodo in ["UVA1-PF (340-400 nm)", "Análise Completa"]:
        uva1_pf = calcular_uva1_pf(df, C)
        resultados["UVA1-PF (340-400 nm)"] = f"{uva1_pf:.2f}"
    
    if metodo in ["CWC", "Análise Completa"]:
        cwc = calcular_cwc(df)
        resultados["CWC"] = f"{cwc:.1f} nm {'✅' if cwc >= 370 else '⚠️'}"
    
    if metodo in ["Método Rápido (Mansur)", "Análise Completa"]:
        spf_mansur = calcular_mansur(df)
        resultados["SPF (Mansur)"] = f"{spf_mansur:.2f}"
    
    if metodo == "Análise Completa":
        absorbancia_uva_longo = calcular_absorbancia_media(df[df['Comprimento de Onda (nm)'] >= 370])
        resultados["Absorbância UVA-Longo (370-400 nm)"] = f"{absorbancia_uva_longo:.3f} {'✅' if absorbancia_uva_longo >= 0.8 else '⚠️'}"
        resultados["Razão UVA/SPF"] = f"{(float(resultados['UVA-PF'].split()[0])/float(resultados['SPF (ISO 24443)'].split()[0])):.2f} {'✅' if (float(resultados['UVA-PF'].split()[0])/float(resultados['SPF (ISO 24443)'].split()[0])) >= 1/3 else '⚠️'}"

    # Exibição dos resultados
    st.subheader("📊 Resultados")
    for parametro, valor in resultados.items():
        st.metric(parametro, valor)

    # Gráficos comparativos
    st.subheader("📈 Comparação entre Métodos")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    metodos = []
    valores = []
    
    if 'SPF (ISO 24443)' in resultados:
        metodos.append("ISO 24443")
        valores.append(float(resultados["SPF (ISO 24443)"].split()[0]))
    
    if 'SPF (Mansur)' in resultados:
        metodos.append("Mansur (1986)")
        valores.append(float(resultados["SPF (Mansur)"].split()[0]))
    
    if metodos:
        ax.bar(metodos, valores, color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel('Valor de SPF')
        ax.set_title('Comparação entre Métodos de Cálculo de SPF')
        
        for i, v in enumerate(valores):
            ax.text(i, v + 1, f"{v:.2f}", ha='center')
        
        st.pyplot(fig)

    # Exportar relatório
    st.subheader("📄 Exportar Resultados")
    if st.button("Gerar Relatório PDF"):
        pdf_bytes = gerar_relatorio(resultados)
        st.download_button(
            label="⬇️ Download Relatório",
            data=pdf_bytes,
            file_name="relatorio_fotoprotecao.pdf",
            mime="application/pdf"
        )

# Seção de ajuda
with st.expander("ℹ️ Instruções Detalhadas"):
    st.markdown("""
    ### 📝 Como Preparar Seus Dados
    **Formato do Excel:**
    - **UVB (290-320 nm):**  
      ```Comprimento de Onda (nm) | Absorbância | E(λ) | I(λ)```
    - **UVA (320-400 nm):**  
      ```Comprimento de Onda (nm) | Absorbância | P(λ) | I(λ)```
    
    **Para dados brutos:**  
    Adicione colunas `Transmitância_amostra` e `Transmitância_branco`.

    ### 🔍 Métodos Científicos
    1. **SPF (ISO 24443):**  
       - Exige SPF in vivo para ajuste da constante C  
    2. **UVA-PF:**  
       - Calculado de 320-400 nm  
    3. **UVA1-PF:**  
       - Subfaixa de 340-400 nm  
    4. **CWC:**  
       - Comprimento onde 90% da absorbância está acumulada  
    5. **Mansur:**  
       - Método rápido para estimativa de SPF  
    """)

# Rodapé científico
st.markdown("---")
st.caption("""
**Referências:**  
1. ISO 24443:2012 - Determination of sunscreen UVA photoprotection in vitro  
2. Mansur et al. (1986) - Determinação do Fator de Proteção Solar por Espectrofotometria  
3. HPC Today (2024) - In vitro method for UVA1, long UVA or ultra-long UVA claiming  
""")
