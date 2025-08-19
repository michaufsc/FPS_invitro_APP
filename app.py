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

# Função para carregar e validar dados - CORRIGIDA
def load_and_validate_data(uploaded_file, data_type="pre_irradiation"):
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
        if data_type == "pre_irradiation":
            required_cols = ['Comprimento de Onda', 'E(λ)', 'I(λ)', 'A0i(λ)']
        else:
            required_cols = ['Comprimento de Onda', 'P(λ)', 'I(λ)', 'Ai(λ)']
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Colunas faltando: {', '.join(missing_cols)}")
            
        return df, None
        
    except Exception as e:
        return None, str(e)

# Função para salvar análise no histórico
def save_to_history(analysis_name, results, timestamp):
    """Salva os resultados da análise no histórico"""
    st.session_state.analysis_history.append({
        'name': analysis_name,
        'results': results,
        'timestamp': timestamp,
        'data_preview': {
            'spf': results.get('spf', 0),
            'spf_adjusted': results.get('spf_adjusted', 0),
            'C_value': results.get('C_value', 0)
        }
    })
    
    # Manter apenas as últimas 10 análises
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history.pop(0)

# Função para criar gráficos avançados
def create_advanced_plots(df, plot_type="absorbance"):
    """Cria visualizações avançadas dos dados"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if plot_type == "absorbance":
        ax.plot(df['Comprimento de Onda'], df['A0i(λ)'], 
               label='Absorbância', linewidth=2.5, color='#1f77b4')
        ax.set_ylabel('Absorbância')
        ax.set_title("Espectro de Absorbância")
        
    elif plot_type == "weighting_functions":
        # Absorbância
        ax.plot(df['Comprimento de Onda'], df['A0i(λ)'], 
               label='Absorbância', linewidth=2.5, color='#1f77b4')
        
        # E(λ) - normalizado para visualização
        if 'E(λ)' in df.columns:
            e_normalized = df['E(λ)'] / df['E(λ)'].max() * df['A0i(λ)'].max() * 0.8
            ax.plot(df['Comprimento de Onda'], e_normalized, 
                   label='E(λ) - Eritema (normalizado)', linewidth=1.5, color='#ff7f0e', linestyle='--')
        
        # I(λ) - normalizado para visualização  
        if 'I(λ)' in df.columns:
            i_normalized = df['I(λ)'] / df['I(λ)'].max() * df['A0i(λ)'].max() * 0.6
            ax.plot(df['Comprimento de Onda'], i_normalized,
                   label='I(λ) - Intensidade (normalizado)', linewidth=1.5, color='#2ca02c', linestyle=':')
        
        ax.set_ylabel('Valores Normalizados')
        ax.set_title("Espectro de Absorbância e Funções de Ponderação")
    
    ax.set_xlabel('Comprimento de Onda (nm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

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
                   ["Cálculo SPF", "Análise UVA", "Métricas Avançadas", "Comparativo", "Explicação das Equações"])
    
    st.markdown("---")
    st.info("""
    **Instruções:**
    1. Faça upload dos dados
    2. Configure os parâmetros
    3. Visualize os resultados
    """)
    
    # Avisos de dependências
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly não está instalado. Para gráficos interativos: `pip install plotly`")

# Página principal baseada na seleção
if page == "Cálculo SPF":
    st.header("🔍 Cálculo do Fator de Proteção Solar (SPF)")
    
    # Upload do arquivo
    uploaded_file = st.file_uploader("📤 Carregue dados pré-irradiação (Excel/CSV)", 
                                   type=["xlsx", "csv"], 
                                   key="spf_upload")
    
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
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")
        
        df, error = load_and_validate_data(uploaded_file, "pre_irradiation")
        
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
                                    value=30.0, step=0.1,
                                    help="Valor do SPF determinado em testes in vivo")
                
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
                    st.metric("Coeficiente C", f"{C_adjusted:.4f}",
                             help="Coeficiente que ajusta o SPF in vitro para in vivo")
                with col3:
                    st.metric("SPF Ajustado", f"{SPF_adjusted:.2f}", 
                             f"{SPF_adjusted - SPF_label:+.2f}",
                             help="SPF calculado após aplicação do coeficiente C")
                
                # Visualização gráfica
                st.subheader("📈 Visualização dos Dados")
                
                plot_type = st.radio("Tipo de visualização:", 
                                    ["Absorbância", "Funções de Ponderação"], 
                                    horizontal=True)
                
                if plot_type == "Absorbância":
                    fig = create_advanced_plots(df, "absorbance")
                else:
                    fig = create_advanced_plots(df, "weighting_functions")
                
                st.pyplot(fig)
                
                # Botão para salvar análise
                analysis_name = st.text_input("💾 Nome para salvar esta análise:", 
                                           value=f"Análise_{datetime.now().strftime('%Y%m%d_%H%M')}")
                
                if st.button("💾 Salvar Análise"):
                    results = {
                        'spf': spf,
                        'spf_adjusted': SPF_adjusted,
                        'C_value': C_adjusted,
                        'spf_labelled': SPF_label
                    }
                    save_to_history(analysis_name, results, datetime.now())
                    st.success(f"✅ Análise '{analysis_name}' salva no histórico!")
                
            except Exception as calc_error:
                st.error(f"❌ Erro no cálculo: {calc_error}")

elif page == "Análise UVA":
    st.header("🌞 Análise de Proteção UVA")
    st.info("Funcionalidade em desenvolvimento...")
    
elif page == "Métricas Avançadas":
    st.header("🔬 Métricas Avançadas")
    
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data['pre_irrad'] is not None:
        df = st.session_state.uploaded_data['pre_irrad']
        
        # Critical Wavelength
        cw = calculate_critical_wavelength(df)
        
        # Exibição dos resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Comprimento de Onda Crítico (nm)", f"{cw:.1f}",
                     "Bom (≥370 nm)" if cw >= 370 else "Abaixo do recomendado")
        
        # Visualização do Critical Wavelength
        st.subheader("📊 Análise do Comprimento de Onda Crítico")
        
        df_uv = df[(df['Comprimento de Onda'] >= 290) & 
                  (df['Comprimento de Onda'] <= 400)].copy()
        
        wavelengths = df_uv['Comprimento de Onda'].to_numpy()
        absorbance = df_uv['A0i(λ)'].to_numpy()
        
        # Calcular área cumulativa
        cumulative_area = np.zeros_like(absorbance)
        for i in range(1, len(absorbance)):
            segment_area = (absorbance[i] + absorbance[i-1])/2 * (wavelengths[i] - wavelengths[i-1])
            cumulative_area[i] = cumulative_area[i-1] + segment_area
        
        total_area = cumulative_area[-1]
        target_area = 0.9 * total_area
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Absorbância
        ax.plot(wavelengths, absorbance, label='Absorbância', linewidth=2.5, color='#1f77b4')
        ax.set_xlabel('Comprimento de Onda (nm)')
        ax.set_ylabel('Absorbância', color='#1f77b4')
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        
        # Área cumulativa
        ax2 = ax.twinx()
        ax2.plot(wavelengths, cumulative_area, label='Área Cumulativa', linewidth=2, color='#ff7f0e', linestyle='--')
        ax2.set_ylabel('Área Cumulativa', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        # Linha do Critical Wavelength
        ax.axvline(x=cw, color='red', linestyle=':', linewidth=2, label=f'λc = {cw:.1f} nm')
        ax2.axhline(y=target_area, color='green', linestyle=':', linewidth=2, label='90% da área total')
        
        ax.set_title("Análise do Comprimento de Onda Crítico")
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
    else:
        st.warning("Por favor, carregue os dados na aba 'Cálculo SPF' primeiro")

elif page == "Comparativo":
    st.header("📊 Comparativo entre Análises")
    
    if not st.session_state.analysis_history:
        st.info("Nenhuma análise salva no histórico ainda.")
    else:
        # Criar DataFrame para comparação
        comparison_data = []
        for analysis in st.session_state.analysis_history:
            comparison_data.append({
                'Análise': analysis['name'],
                'Data': analysis['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'SPF': analysis['data_preview']['spf'],
                'SPF Ajustado': analysis['data_preview']['spf_adjusted'],
                'Coeficiente C': analysis['data_preview']['C_value']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison.style.highlight_max(axis=0, color='#90EE90'), use_container_width=True)
        
        # Gráfico comparativo
        if len(st.session_state.analysis_history) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            analyses = [a['name'] for a in st.session_state.analysis_history]
            spf_values = [a['data_preview']['spf'] for a in st.session_state.analysis_history]
            adjusted_values = [a['data_preview']['spf_adjusted'] for a in st.session_state.analysis_history]
            
            x = np.arange(len(analyses))
            width = 0.35
            
            ax.bar(x - width/2, spf_values, width, label='SPF In Vitro', alpha=0.8)
            ax.bar(x + width/2, adjusted_values, width, label='SPF Ajustado', alpha=0.8)
            
            ax.set_xlabel('Análises')
            ax.set_ylabel('Valor do SPF')
            ax.set_title('Comparação entre Análises')
            ax.set_xticks(x)
            ax.set_xticklabels(analyses, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

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
    
    st.markdown("""
    ### 3. Comprimento de Onda Crítico ($\lambda_c$)
    """)
    
    st.latex(r'''
    \lambda_c = \min \left\{ \lambda \middle| \int_{290}^{\lambda} A(\lambda)  d\lambda \geq 0.9 \times \int_{290}^{400} A(\lambda)  d\lambda \right\}
    ''')
    
    st.markdown("""
    **Interpretação:**
    - $\lambda_c \geq 370$ nm indica boa proteção UVA
    - Valores abaixo de 370 nm sugerem proteção UVA insuficiente
    """)

# Rodapé
st.markdown("---")
st.markdown("""
**🔬 Desenvolvido para análise de proteção solar**  
*Sistema compatível com diversos formatos de dados*
""")
