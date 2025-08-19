import streamlit as st

# Configuração da página DEVE SER A PRIMEIRA INSTRUÇÃO
st.set_page_config(
    page_title="Análise de Proteção Solar",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agora os outros imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from datetime import datetime

# Verificar e importar Plotly com fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Sistema de sessão para armazenar dados
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {
        'pre_irrad': None,
        'post_irrad': None,
        'metadata': {}
    }

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Função para mapear nomes de colunas - CORRIGIDA
def map_column_names(df):
    """Mapeia nomes de colunas para um formato padrão"""
    column_mapping = {}
    
    # Mapear comprimento de onda - CORREÇÃO PRINCIPAL
    for col in df.columns:
        lower_col = col.lower().strip()
        # Verifica várias possibilidades
        if any(word in lower_col for word in ['comprimento', 'onda', 'wavelength', 'lambda', 'nm']):
            column_mapping[col] = 'Comprimento de Onda'
            break
    else:
        # Se não encontrou, usa a primeira coluna
        column_mapping[df.columns[0]] = 'Comprimento de Onda'
    
    # Mapear absorbância
    absorbance_aliases = ['absorbancia', 'absorvancia', 'absorbância', 'absorvância', 'abs', 'a0i']
    for col in df.columns:
        lower_col = col.lower()
        if any(alias in lower_col for alias in absorbance_aliases):
            column_mapping[col] = 'A0i(λ)'
            break
    else:
        # Se não encontrou, tenta encontrar por padrão
        for col in df.columns:
            lower_col = col.lower()
            if 'abs' in lower_col:
                column_mapping[col] = 'A0i(λ)'
                break
    
    # Mapear E(λ)
    eritema_aliases = ['e(λ)', 'e(lambda)', 'eritema', 'erythema', 'e(', 'e ']
    for col in df.columns:
        lower_col = col.lower()
        if any(alias in lower_col for alias in eritema_aliases):
            column_mapping[col] = 'E(λ)'
            break
    
    # Mapear I(λ)
    intensity_aliases = ['i(λ)', 'i(lambda)', 'intensidade', 'intensity', 'i ']
    for col in df.columns:
        lower_col = col.lower()
        if any(alias in lower_col for alias in intensity_aliases):
            column_mapping[col] = 'I(λ)'
            break
    
    return column_mapping

# Funções de cálculo
def calculate_spf(df):
    """Calcula SPF in vitro conforme Equação 1"""
    d_lambda = 1
    E = df['E(λ)'].to_numpy()
    I = df['I(λ)'].to_numpy()
    A0i = df['A0i(λ)'].to_numpy()
    T = 10 ** (-A0i)
    
    numerator = np.sum(E * I * d_lambda)
    denominator = np.sum(E * I * T * d_lambda)
    
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    return numerator / denominator

def calculate_adjusted_spf(df, C):
    """Calcula SPF ajustado conforme Equação 2"""
    d_lambda = 1
    E = df['E(λ)'].to_numpy()
    I = df['I(λ)'].to_numpy()
    A0i = df['A0i(λ)'].to_numpy()
    
    numerator = np.sum(E * I * d_lambda)
    denominator = np.sum(E * I * (10 ** (-A0i * C)) * d_lambda)
    
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    return numerator / denominator

def calculate_uva_pf(df, C):
    """Calcula UVA-PF"""
    d_lambda = 1
    P = df['P(λ)'].to_numpy()
    I = df['I(λ)'].to_numpy()
    A0i = df['A0i(λ)'].to_numpy()
    
    numerator = np.sum(P * I * d_lambda)
    denominator = np.sum(P * I * (10 ** (-A0i * C)) * d_lambda)
    
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    return numerator / denominator

def calculate_critical_wavelength(df):
    """Calcula Critical Wavelength"""
    df_uv = df[(df['Comprimento de Onda'] >= 290) & 
              (df['Comprimento de Onda'] <= 400)].copy()
    
    wavelengths = df_uv['Comprimento de Onda'].to_numpy()
    absorbance = df_uv['A0i(λ)'].to_numpy()
    
    total_area = np.trapz(absorbance, wavelengths)
    target_area = 0.9 * total_area
    
    cumulative_area = 0
    for i, (wl, abs_val) in enumerate(zip(wavelengths, absorbance)):
        if i == 0:
            continue
        segment_area = (abs_val + absorbance[i-1])/2 * (wl - wavelengths[i-1])
        cumulative_area += segment_area
        
        if cumulative_area >= target_area:
            return wl
    
    return 400

# Função para carregar e validar dados
def load_and_validate_data(uploaded_file):
    """Carrega e valida os dados do arquivo com mapeamento de colunas"""
    try:
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        df.columns = df.columns.str.strip()
        
        # Mapear nomes de colunas
        column_mapping = map_column_names(df)
        df = df.rename(columns=column_mapping)
        
        # Verificar colunas necessárias
        required_cols = ['Comprimento de Onda', 'E(λ)', 'I(λ)', 'A0i(λ)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Colunas faltando após mapeamento: {', '.join(missing_cols)}")
            
        return df, None
        
    except Exception as e:
        return None, str(e)

# LOGOS NO TOPO
col1, col2 = st.columns([1, 0.5])
with col1:
    st.image("https://via.placeholder.com/200x80/FF9900/000000?text=MEU+LAB", width=200)
with col2:
    st.image("https://via.placeholder.com/200x80/0066CC/FFFFFF?text=UFSC", width=200)

# TÍTULO PRINCIPAL
st.title("🌞 Análise Completa de Proteção Solar")

# Menu lateral para navegação
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/FF9900/000000?text=LOGO", width=150)
    st.title("Navegação")
    page = st.radio("Selecione a página:", 
                   ["Cálculo SPF", "Análise UVA", "Métricas Avançadas", "Explicação das Equações"])
    
    st.markdown("---")
    st.info("""
    **Instruções:**
    1. Faça upload dos dados
    2. Configure os parâmetros
    3. Visualize os resultados
    """)

# Página principal baseada na seleção
if page == "Cálculo SPF":
    st.header("🔍 Cálculo do Fator de Proteção Solar (SPF)")
    
    # Upload do arquivo
    uploaded_file = st.file_uploader("Carregue dados pré-irradiação (Excel/CSV)", 
                                   type=["xlsx", "csv"], 
                                   key="spf_upload")
    
    if uploaded_file:
        df, error = load_and_validate_data(uploaded_file)
        
        if error:
            st.error(f"Erro ao processar arquivo: {error}")
            st.info("""
            **Formato esperado:** Seu arquivo deve conter colunas para:
            - Comprimento de Onda (nm)
            - E(λ) [Eritema]
            - I(λ) [Intensidade]
            - Absorbância [pode ser chamada de Absorbancia, Absorvancia, etc.]
            """)
        else:
            # Mostrar preview dos dados
            st.subheader("📋 Dados processados (primeiras linhas)")
            st.dataframe(df.head(), use_container_width=True)
            
            # Cálculo do SPF
            try:
                spf = calculate_spf(df)
                st.success(f"📊 SPF in vitro calculado: {spf:.2f}")
                
                # Ajuste do SPF
                st.subheader("🔧 Ajuste do SPF")
                SPF_label = st.number_input("SPF rotulado (in vivo)", 
                                         min_value=1.0, value=30.0, step=0.1)
                
                # Otimização para encontrar C
                def error_function(C):
                    return abs(calculate_adjusted_spf(df, C) - SPF_label)
                
                result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                C_adjusted = result.x
                SPF_adjusted = calculate_adjusted_spf(df, C_adjusted)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Coeficiente de Ajuste (C)", f"{C_adjusted:.4f}")
                with col2:
                    st.metric("SPF Ajustado", f"{SPF_adjusted:.2f}")
                
                # Visualização
                st.subheader("📈 Visualização dos Dados")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['Comprimento de Onda'], df['A0i(λ)'], label='Absorbância', linewidth=2)
                ax.plot(df['Comprimento de Onda'], df['E(λ)'], label='E(λ) - Eritema', linewidth=2)
                ax.plot(df['Comprimento de Onda'], df['I(λ)'], label='I(λ) - Intensidade', linewidth=2)
                ax.set_xlabel('Comprimento de Onda (nm)')
                ax.set_ylabel('Valores')
                ax.set_title("Dados Pré-Irradiação")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
            except ValueError as e:
                st.error(f"Erro no cálculo: {str(e)}")

elif page == "Análise UVA":
    st.header("🌞 Análise de Proteção UVA")
    st.info("Funcionalidade em desenvolvimento...")

elif page == "Métricas Avançadas":
    st.header("🔬 Métricas Avançadas")
    st.info("Funcionalidade em desenvolvimento...")

elif page == "Explicação das Equações":
    st.header("📚 Explicação das Equações Matemáticas")
    
    st.markdown("""
    ## 📊 Equações Principais
    
    ### 1. Cálculo do SPF in vitro
    """)
    
    st.latex(r'''
    SPF = \frac{\sum_{290}^{400} E(\lambda) \times I(\lambda) \times \Delta\lambda}
    {\sum_{290}^{400} E(\lambda) \times I(\lambda) \times T(\lambda) \times \Delta\lambda}
    ''')
    
    st.markdown("""
    **Onde:**
    - $E(\lambda)$ = Eficiência relativa de produção de eritema
    - $I(\lambda)$ = Intensidade spectral da luz solar  
    - $T(\lambda)$ = Transmitância da amostra ($T = 10^{-A(\lambda)}$)
    - $A(\lambda)$ = Absorbância da amostra
    - $\Delta\lambda$ = Intervalo entre comprimentos de onda (1 nm)
    """)
    
    st.markdown("""
    ### 2. SPF Ajustado com Coeficiente C
    """)
    
    st.latex(r'''
    SPF_{\text{ajustado}} = \frac{\sum E(\lambda) \times I(\lambda) \times \Delta\lambda}
    {\sum E(\lambda) \times I(\lambda) \times 10^{-A(\lambda) \times C} \times \Delta\lambda}
    ''')
    
    st.markdown("""
    **Onde:**
    - $C$ = Coeficiente de ajuste que correlaciona o SPF in vitro com o SPF in vivo
    """)

# Rodapé
st.markdown("---")
st.markdown("""
**Desenvolvido para análise de proteção solar**  
*Sistema compatível com diversos formatos de dados*
""")

# Avisos de dependências
if not PLOTLY_AVAILABLE:
    st.sidebar.warning("Plotly não está instalado. Para gráficos interativos: `pip install plotly`")
