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

# Sistema de sessão para armazenar dados
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {
        'pre_irrad': None,
        'post_irrad': None,
        'metadata': {}
    }

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
    for col in df.columns:
        lower_col = col.lower().strip()
        if any(word in lower_col for word in ['absorbancia', 'absorvancia', 'absorbância', 'absorvância', 'abs']):
            column_mapping[col] = 'A0i(λ)'
            break
    
    # Mapear E(λ)
    for col in df.columns:
        lower_col = col.lower().strip()
        if any(word in lower_col for word in ['e(λ)', 'e(lambda)', 'eritema', 'erythema', 'e(']):
            column_mapping[col] = 'E(λ)'
            break
    
    # Mapear I(λ)
    for col in df.columns:
        lower_col = col.lower().strip()
        if any(word in lower_col for word in ['i(λ)', 'i(lambda)', 'intensidade', 'intensity', 'i(']):
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

# Função para carregar e validar dados - CORRIGIDA
def load_and_validate_data(uploaded_file):
    """Carrega e valida os dados do arquivo com mapeamento de colunas"""
    try:
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Mostrar colunas originais
        st.write("🔍 **Colunas originais:**", list(df.columns))
        
        # Mapear nomes de colunas
        column_mapping = map_column_names(df)
        st.write("🔄 **Mapeamento:**", column_mapping)
        
        df = df.rename(columns=column_mapping)
        st.write("✅ **Colunas após mapeamento:**", list(df.columns))
        
        # Verificar se todas as colunas necessárias estão presentes
        required_cols = ['Comprimento de Onda', 'E(λ)', 'I(λ)', 'A0i(λ)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Colunas faltando: {', '.join(missing_cols)}")
            
        return df, None
        
    except Exception as e:
        return None, str(e)

# Interface principal
st.title("🌞 Análise de Proteção Solar")
st.markdown("---")

# Upload do arquivo
uploaded_file = st.file_uploader("📤 Carregue seu arquivo de dados (Excel/CSV)", 
                               type=["xlsx", "csv"])

if uploaded_file:
    st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
    
    # Debug: mostrar conteúdo bruto do arquivo
    with st.expander("🔍 Visualizar conteúdo original do arquivo"):
        try:
            if uploaded_file.name.endswith('xlsx'):
                raw_df = pd.read_excel(uploaded_file)
            else:
                raw_df = pd.read_csv(uploaded_file)
            st.write("📋 **Conteúdo original (primeiras 5 linhas):**")
            st.dataframe(raw_df.head())
            st.write(f"📊 **Total de linhas:** {len(raw_df)}")
            st.write("🔤 **Colunas originais:**", list(raw_df.columns))
            st.write("📝 **Tipos de dados:**")
            st.write(raw_df.dtypes)
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
    
    df, error = load_and_validate_data(uploaded_file)
    
    if error:
        st.error(f"❌ Erro ao processar arquivo: {error}")
        
        # Tentativa alternativa de leitura
        st.info("🔄 Tentando leitura alternativa...")
        try:
            if uploaded_file.name.endswith('xlsx'):
                alt_df = pd.read_excel(uploaded_file)
            else:
                alt_df = pd.read_csv(uploaded_file)
            
            st.write("📋 **Estrutura do arquivo:**")
            st.dataframe(alt_df.head(3))
            
            # Tentativa manual de mapeamento
            st.write("🎯 **Sugestão de mapeamento manual:**")
            if len(alt_df.columns) >= 4:
                manual_mapping = {
                    alt_df.columns[0]: 'Comprimento de Onda',
                    alt_df.columns[1]: 'E(λ)',
                    alt_df.columns[2]: 'I(λ)', 
                    alt_df.columns[3]: 'A0i(λ)'
                }
                st.write("Mapeamento sugerido:", manual_mapping)
                
        except Exception as alt_error:
            st.error(f"Erro na leitura alternativa: {alt_error}")
            
    else:
        # Dados processados com sucesso
        st.success("✅ Arquivo processado com sucesso!")
        
        # Mostrar dados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dados processados")
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.subheader("📈 Estatísticas")
            st.metric("Total de pontos", len(df))
            st.metric("Faixa de comprimento de onda", 
                     f"{df['Comprimento de Onda'].min()} - {df['Comprimento de Onda'].max()} nm")
            st.metric("Absorbância máxima", f"{df['A0i(λ)'].max():.3f}")
            st.metric("Absorbância mínima", f"{df['A0i(λ)'].min():.3f}")
        
        # Cálculo do SPF
        try:
            with st.spinner("🧮 Calculando SPF..."):
                spf = calculate_spf(df)
            
            st.success(f"✅ **SPF in vitro calculado:** {spf:.2f}")
            
            # Ajuste do SPF
            st.subheader("🔧 Ajuste do SPF")
            SPF_label = st.slider("SPF rotulado (in vivo)", 
                                min_value=1.0, max_value=100.0, 
                                value=30.0, step=0.1)
            
            with st.spinner("⚙️ Otimizando coeficiente de ajuste..."):
                def error_function(C):
                    return abs(calculate_adjusted_spf(df, C) - SPF_label)
                
                result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                C_adjusted = result.x
                SPF_adjusted = calculate_adjusted_spf(df, C_adjusted)
            
            # Resultados
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("SPF In Vitro", f"{spf:.2f}")
            with col2:
                st.metric("Coeficiente C", f"{C_adjusted:.4f}")
            with col3:
                st.metric("SPF Ajustado", f"{SPF_adjusted:.2f}")
            
            # Visualização gráfica
            st.subheader("📈 Visualização dos Dados")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Comprimento de Onda'], df['A0i(λ)'], 
                   label='Absorbância', linewidth=2, color='blue')
            ax.set_xlabel('Comprimento de Onda (nm)')
            ax.set_ylabel('Absorbância')
            ax.set_title("Espectro de Absorbância")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        except Exception as calc_error:
            st.error(f"❌ Erro no cálculo: {calc_error}")

else:
    st.info("📝 Por favor, carregue um arquivo Excel ou CSV para começar a análise")

# Rodapé
st.markdown("---")
st.markdown("**🔬 Sistema de Análise de Proteção Solar**")
