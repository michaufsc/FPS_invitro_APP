import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.integrate import cumulative_trapezoid
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

def calcular_spf(df, C=1.0, lambda_min=290, lambda_max=400):
    """Calcula SPF in vitro (Diffey 290-400 nm)"""
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
    """Calcula UVA-PF (ISO 24443, forma simplificada)"""
    df_uva = df[(df['Comprimento de Onda (nm)'] >= 320)].copy()
    num = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'], df_uva['Comprimento de Onda (nm)'])
    den = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'] * 10**(-df_uva['Absorbância'] * C), 
                  df_uva['Comprimento de Onda (nm)'])
    return num / den

def calcular_uva1_pf(df, C=1.0):
    """Calcula UVA1-PF (340-400 nm)"""
    df_uva1 = df[(df['Comprimento de Onda (nm)'] >= 340)].copy()
    num = np.trapz(df_uva1['P(λ)'] * df_uva1['I(λ)'], df_uva1['Comprimento de Onda (nm)'])
    den = np.trapz(df_uva1['P(λ)'] * df_uva1['I(λ)'] * 10**(-df_uva1['Absorbância'] * C), 
                  df_uva1['Comprimento de Onda (nm)'])
    return num / den

def calcular_cwc(df):
    """Calcula Comprimento de Onda Crítico (ISO 24443)"""
    df_faixa = df[(df['Comprimento de Onda (nm)'] >= 290) & 
                 (df['Comprimento de Onda (nm)'] <= 400)].copy()
    
    x = df_faixa['Comprimento de Onda (nm)'].values
    y = df_faixa['Absorbância'].values

    area_total = np.trapz(y, x)
    area_cum = cumulative_trapezoid(y, x, initial=0)

    # Ponto onde atinge 90% da área
    alvo = 0.9 * area_total
    idx = np.where(area_cum >= alvo)[0][0]

    # Interpolação linear
    if idx == 0:
        return x[0]
    frac = (alvo - area_cum[idx-1]) / (area_cum[idx] - area_cum[idx-1])
    lambda_c = x[idx-1] + frac * (x[idx] - x[idx-1])
    return lambda_c

def calcular_mansur(df):
    """Calcula SPF pelo método de Mansur (1986)"""
    df['Contribuição'] = df['Comprimento de Onda (nm)'].map(COEF_MANSUR) * df['Absorbância']
    return 10 * df['Contribuição'].sum()

def calcular_absorbancia_media(df, lmin=370, lmax=400):
    """Calcula absorbância média em uma faixa espectral"""
    faixa = df[(df['Comprimento de Onda (nm)'] >= lmin) & (df['Comprimento de Onda (nm)'] <= lmax)]
    return faixa['Absorbância'].mean() if not faixa.empty else np.nan

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
✔ SPF in vitro (Diffey 290–400 nm)  
✔ UVA-PF (ISO 24443 simplificado)  
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
        "SPF in vitro (Diffey)", 
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
    
    if metodo in ["SPF in vitro (Diffey)", "Análise Completa"]:
        spf_in_vivo = st.number_input("SPF in vivo conhecido (para ajuste):", value=30.0)
        res = opt.minimize_scalar(lambda C: (calcular_spf(df, C)[0] - spf_in_vivo)**2,
                                  bounds=(0.3, 2.0), method='bounded')
        C = res.x
        spf, incerteza = calcular_spf(df, C)
        resultados["SPF in vitro (Diffey)"] = f"{spf:.2f} ± {incerteza:.2f} (C={C:.3f})"
    
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
        absorbancia_uva_longo = calcular_absorbancia_media(df, 370, 400)
        resultados["Absorbância UVA-Longo (370-400 nm)"] = f"{absorbancia_uva_longo:.3f} {'✅' if absorbancia_uva_longo >= 0.8 else '⚠️'}"
        if "SPF in vitro (Diffey)" in resultados and "UVA-PF" in resultados:
            uva_spf_ratio = float(resultados["UVA-PF"].split()[0]) / float(resultados["SPF in vitro (Diffey)"].split()[0])
            resultados["Razão UVA/SPF"] = f"{uva_spf_ratio:.2f} {'✅' if uva_spf_ratio >= 1/3 else '⚠️'}"

    # Exibição dos resultados
    st.subheader("📊 Resultados")
    for parametro, valor in resultados.items():
        st.metric(parametro, valor)

    # Gráficos comparativos
    st.subheader("📈 Comparação entre Métodos")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    metodos = []
    valores = []
    
    if 'SPF in vitro (Diffey)' in resultados:
        metodos.append("Diffey (in vitro)")
        valores.append(float(resultados["SPF in vitro (Diffey)"].split()[0]))
    
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
    - **UVB/UVA (290-400 nm):**  
      ```Comprimento de Onda (nm) | Absorbância | E(λ) | P(λ) | I(λ)```
    
    **Para dados brutos:**  
    Adicione colunas `Transmitância_amostra` e `Transmitância_branco`.

    ### 🔍 Métodos Científicos
    1. **SPF in vitro (Diffey 290–400 nm):**  
       - Ajuste de C via SPF in vivo conhecido
    2. **UVA-PF (ISO 24443, forma simplificada):**  
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
1. Diffey BL (1994) – When should sunscreen be reapplied? J Am Acad Dermatol.  
2. ISO 24443:2012 – Determination of sunscreen UVA photoprotection in vitro.  
3. Mansur et al. (1986) – Determinação do Fator de Proteção Solar por Espectrofotometria.  
4. HPC Today (2024) – In vitro method for UVA1, long UVA or ultra-long UVA claiming.  
""")
