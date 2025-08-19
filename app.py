import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from datetime import datetime

# Configuração da página DEVE SER A PRIMEIRA INSTRUÇÃO
st.set_page_config(
    page_title="Análise de Proteção Solar",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sistema de sessão para armazenar dados
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {
        'pre_irrad': None,
        'post_irrad': None,
        'metadata': {}
    }

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'current_results' not in st.session_state:
    st.session_state.current_results = {}

# Função para mapear nomes de colunas
def map_column_names(df, data_type="pre_irradiation"):
    """Mapeia nomes de colunas para um formato padrão"""
    column_mapping = {}
    
    # Mapear comprimento de onda
    for col in df.columns:
        lower_col = col.lower().strip()
        if any(word in lower_col for word in ['wavelength', 'comprimento', 'onda', 'lambda', 'nm']):
            column_mapping[col] = 'Comprimento de Onda'
            break
    else:
        column_mapping[df.columns[0]] = 'Comprimento de Onda'
    
    # Mapeamento baseado no tipo de dados
    if data_type == "pre_irradiation":
        # Para dados pré-irradiação (SPF)
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['absorbancia', 'absorvancia', 'absorbância', 'absorvância', 'abs', 'a0']):
                column_mapping[col] = 'A0i(λ)'
                break
        
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['e(λ)', 'e(lambda)', 'eritema', 'erythema', 'e(']):
                column_mapping[col] = 'E(λ)'
                break
        
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['i(λ)', 'i(lambda)', 'intensidade', 'intensity', 'i(']):
                column_mapping[col] = 'I(λ)'
                break
                
    else:
        # Para dados pós-irradiação (UVA)
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['a_e', 'absorbancia', 'absorvancia', 'absorbância', 'absorvância', 'abs', 'ai']):
                column_mapping[col] = 'Ai(λ)'
                break
        
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['p', 'p(λ)', 'p(lambda)', 'pigmentacao', 'pigmentação']):
                column_mapping[col] = 'P(λ)'
                break
        
        for col in df.columns:
            lower_col = col.lower().strip()
            if any(word in lower_col for word in ['i', 'i(λ)', 'i(lambda)', 'intensidade', 'intensity']):
                column_mapping[col] = 'I(λ)'
                break
    
    return column_mapping

# Funções de cálculo SPF
def calculate_spf(df):
    """Calcula SPF in vitro"""
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
    """Calcula SPF ajustado"""
    d_lambda = 1
    E = df['E(λ)'].to_numpy()
    I = df['I(λ)'].to_numpy()
    A0i = df['A0i(λ)'].to_numpy()
    
    numerator = np.sum(E * I * d_lambda)
    denominator = np.sum(E * I * (10 ** (-A0i * C)) * d_lambda)
    
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    return numerator / denominator

# FUNÇÃO UVA-PF
def calculate_uva_pf(df, C):
    """Calcula UVA Protection Factor"""
    d_lambda = 1
    
    # Obter os arrays
    P = df['P(λ)'].to_numpy()
    I = df['I(λ)'].to_numpy()
    A_e = df['Ai(λ)'].to_numpy()
    
    # Cálculo do UVA-PF
    numerador = np.sum(P * I * d_lambda)
    denominador = np.sum(P * I * 10**(-A_e * C) * d_lambda)
    
    if denominador == 0:
        raise ValueError("Denominador não pode ser zero")
    
    return numerador / denominador

# Funções adicionais para UVA
def calculate_critical_wavelength(df):
    """Calcula Critical Wavelength"""
    df_uv = df[(df['Comprimento de Onda'] >= 290) & 
              (df['Comprimento de Onda'] <= 400)].copy()
    
    wavelengths = df_uv['Comprimento de Onda'].to_numpy()
    absorbance = df_uv['Ai(λ)'].to_numpy()
    
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

def calculate_uva_uv_ratio(df):
    """Calcula Razão UVA/UV"""
    mask_uva = (df['Comprimento de Onda'] >= 320) & (df['Comprimento de Onda'] <= 400)
    uva_area = np.trapz(df[mask_uva]['Ai(λ)'], df[mask_uva]['Comprimento de Onda'])
    
    mask_uv = (df['Comprimento de Onda'] >= 290) & (df['Comprimento de Onda'] <= 400)
    uv_area = np.trapz(df[mask_uv]['Ai(λ)'], df[mask_uv]['Comprimento de Onda'])
    
    return uva_area / uv_area if uv_area != 0 else 0

# Função para carregar e validar dados
def load_and_validate_data(uploaded_file, data_type="pre_irradiation"):
    """Carrega e valida os dados do arquivo"""
    try:
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Mostrar colunas originais
        st.write("🔍 **Colunas originais:**", list(df.columns))
        
        # Mapear nomes de colunas
        column_mapping = map_column_names(df, data_type)
        st.write("🔄 **Mapeamento:**", column_mapping)
        
        df = df.rename(columns=column_mapping)
        st.write("✅ **Colunas após mapeamento:**", list(df.columns))
        
        # Verificar colunas necessárias
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

# Função para salvar análise
def save_to_history(analysis_name, results, timestamp):
    """Salva análise no histórico"""
    st.session_state.analysis_history.append({
        'name': analysis_name,
        'results': results,
        'timestamp': timestamp,
        'data_preview': {
            'spf': results.get('spf', 0),
            'spf_adjusted': results.get('spf_adjusted', 0),
            'C_value': results.get('C_value', 0),
            'uva_pf': results.get('uva_pf', 0)
        }
    })
    
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history.pop(0)

# Interface principal
st.title("🌞 Análise Completa de Proteção Solar")

# Menu lateral
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/FF9900/000000?text=LOGO", width=150)
    st.title("Navegação")
    page = st.radio("Selecione a página:", 
                   ["Cálculo SPF", "Análise UVA", "Métricas Avançadas", "Comparativo", "Explicação das Equações"])
    
    st.markdown("---")
    st.info("""
    **Instruções:**
    1. Calcule SPF primeiro
    2. Use o coeficiente C para UVA
    3. Analise os resultados
    """)

# Páginas
if page == "Cálculo SPF":
    st.header("🔍 Cálculo do Fator de Proteção Solar (SPF)")
    
    uploaded_file = st.file_uploader("📤 Carregue dados pré-irradiação (Excel/CSV)", 
                                   type=["xlsx", "csv"], 
                                   key="spf_upload")
    
    if uploaded_file:
        df, error = load_and_validate_data(uploaded_file, "pre_irradiation")
        
        if error:
            st.error(f"❌ Erro: {error}")
        else:
            st.success("✅ Dados processados!")
            st.dataframe(df.head())
            
            try:
                with st.spinner("🧮 Calculando SPF..."):
                    spf = calculate_spf(df)
                
                st.success(f"✅ **SPF in vitro:** {spf:.2f}")
                
                SPF_label = st.slider("SPF rotulado (in vivo)", 
                                    min_value=1.0, max_value=100.0, 
                                    value=30.0, step=0.1)
                
                with st.spinner("⚙️ Otimizando coeficiente C..."):
                    def error_function(C):
                        return abs(calculate_adjusted_spf(df, C) - SPF_label)
                    
                    result = opt.minimize_scalar(error_function, bounds=(0.5, 1.6), method='bounded')
                    C_adjusted = result.x
                    SPF_adjusted = calculate_adjusted_spf(df, C_adjusted)
                
                # Salvar resultados
                st.session_state.current_results = {
                    'spf': spf,
                    'spf_adjusted': SPF_adjusted,
                    'C_value': C_adjusted,
                    'spf_labelled': SPF_label
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("SPF In Vitro", f"{spf:.2f}")
                with col2:
                    st.metric("Coeficiente C", f"{C_adjusted:.4f}")
                with col3:
                    st.metric("SPF Ajustado", f"{SPF_adjusted:.2f}")
                
                # Botão para salvar
                analysis_name = st.text_input("💾 Nome para salvar:", 
                                           value=f"SPF_{datetime.now().strftime('%H%M')}")
                
                if st.button("💾 Salvar Análise"):
                    save_to_history(analysis_name, st.session_state.current_results, datetime.now())
                    st.success("✅ Análise salva!")
                    
            except Exception as e:
                st.error(f"❌ Erro no cálculo: {e}")

elif page == "Análise UVA":
    st.header("🌞 Análise de Proteção UVA")
    
    # Verificar se já calculou SPF
    if 'C_value' not in st.session_state.current_results:
        st.warning("⚠️ Calcule o SPF primeiro para obter o coeficiente C")
        C_default = 0.8
        spf_default = 30.0
    else:
        C_default = st.session_state.current_results['C_value']
        spf_default = st.session_state.current_results['spf_labelled']
        st.success(f"✅ Coeficiente C disponível: {C_default:.4f}")
    
    # Upload dados UVA
    uva_file = st.file_uploader("📤 Carregue dados UVA (Excel/CSV) - Colunas: wavelength, P, I, A_e", 
                              type=["xlsx", "csv"], 
                              key="uva_upload")
    
    if uva_file:
        df_uva, error = load_and_validate_data(uva_file, "post_irradiation")
        
        if error:
            st.error(f"❌ Erro: {error}")
        else:
            st.success("✅ Dados UVA processados!")
            st.dataframe(df_uva.head())
            
            # Mostrar mapeamento específico para o usuário
            st.info("""
            **📋 Mapeamento das colunas:**
            - `wavelength` → `Comprimento de Onda`
            - `P` → `P(λ)`
            - `I` → `I(λ)`
            - `A_e` → `Ai(λ)`
            """)
            
            # Configurações UVA
            col1, col2 = st.columns(2)
            with col1:
                C_uva = st.number_input("Coeficiente C", 
                                      min_value=0.1, max_value=2.0, 
                                      value=float(C_default), step=0.01,
                                      help="Coeficiente de ajuste para UVA")
            with col2:
                spf_label = st.number_input("SPF Rotulado", 
                                         min_value=1.0, value=float(spf_default), step=0.1)
            
            if st.button("🧮 Calcular UVA-PF", type="primary"):
                with st.spinner("Calculando UVA-PF..."):
                    try:
                        uva_pf = calculate_uva_pf(df_uva, C_uva)
                        
                        # Calcular métricas adicionais
                        critical_wl = calculate_critical_wavelength(df_uva)
                        uva_ratio = calculate_uva_uv_ratio(df_uva)
                        uva_spf_ratio = uva_pf / spf_label if spf_label != 0 else 0
                        
                        # Resultados
                        st.subheader("📊 Resultados UVA")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("UVA-PF", f"{uva_pf:.2f}")
                        with col2:
                            status = "✅ Bom" if critical_wl >= 370 else "⚠️ Baixo"
                            st.metric("λ Crítico", f"{critical_wl:.1f} nm", status)
                        with col3:
                            status = "✅ Bom" if uva_spf_ratio >= 0.33 else "⚠️ Baixo"
                            st.metric("UVA/SPF", f"{uva_spf_ratio:.2f}", status)
                        with col4:
                            st.metric("Razão UVA/UV", f"{uva_ratio:.2f}")
                        
                        # Avaliação
                        if critical_wl >= 370 and uva_spf_ratio >= 0.33:
                            st.success("✅ **PRODUTO CONFORME** - Atende aos requisitos UVA")
                        else:
                            st.warning("⚠️ **PRODUTO NÃO CONFORME** - Verifique os requisitos UVA")
                        
                        # Gráfico
                        st.subheader("📈 Espectro UVA")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Dados UVA (320-400 nm)
                        df_uva_region = df_uva[(df_uva['Comprimento de Onda'] >= 320) & 
                                             (df_uva['Comprimento de Onda'] <= 400)]
                        
                        ax.plot(df_uva_region['Comprimento de Onda'], df_uva_region['Ai(λ)'], 
                               label='Absorbância UVA', linewidth=2, color='purple')
                        
                        # Marcar região UVA
                        ax.axvspan(320, 400, alpha=0.1, color='purple', label='Região UVA (320-400 nm)')
                        
                        # Marcar critical wavelength
                        ax.axvline(x=critical_wl, color='red', linestyle='--', 
                                  label=f'λ Crítico = {critical_wl:.1f} nm')
                        
                        ax.set_xlabel('Comprimento de Onda (nm)')
                        ax.set_ylabel('Absorbância')
                        ax.set_title('Espectro de Absorbância na Região UVA')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Salvar resultados UVA
                        st.session_state.current_results['uva_pf'] = uva_pf
                        st.session_state.current_results['critical_wavelength'] = critical_wl
                        st.session_state.current_results['uva_spf_ratio'] = uva_spf_ratio
                        
                    except Exception as e:
                        st.error(f"❌ Erro no cálculo UVA: {e}")

elif page == "Métricas Avançadas":
    st.header("🔬 Métricas Avançadas")
    st.info("Visualização detalhada das métricas de proteção solar")
    
    if 'current_results' in st.session_state and 'uva_pf' in st.session_state.current_results:
        st.subheader("📈 Métricas Calculadas")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("UVA-PF", f"{st.session_state.current_results['uva_pf']:.2f}")
        with col2:
            st.metric("λ Crítico", f"{st.session_state.current_results['critical_wavelength']:.1f} nm")
        with col3:
            st.metric("Razão UVA/SPF", f"{st.session_state.current_results['uva_spf_ratio']:.2f}")
        
        # Informações técnicas
        with st.expander("📋 Informações Técnicas"):
            st.markdown("""
            **Critérios de Avaliação:**
            - **λ Crítico ≥ 370 nm**: Proteção UVA adequada
            - **UVA-PF/SPF ≥ 0.33**: Boa relação de proteção UVA/UVB
            - **UVA-PF ≥ 10**: Proteção UVA significativa
            
            **Referências:**
            - ISO 24443:2012 - Determination of sunscreen UVA photoprotection in vitro
            - COLIPA/CTFA/JCIA: International sun protection factor test method
            """)
    else:
        st.warning("⚠️ Execute a análise UVA primeiro para ver métricas avançadas")

elif page == "Comparativo":
    st.header("📊 Comparativo entre Análises")
    
    if st.session_state.analysis_history:
        # Criar DataFrame comparativo
        comparison_data = []
        for analysis in st.session_state.analysis_history:
            comparison_data.append({
                'Análise': analysis['name'],
                'Data': analysis['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'SPF': f"{analysis['data_preview']['spf']:.2f}",
                'SPF Ajustado': f"{analysis['data_preview']['spf_adjusted']:.2f}",
                'Coeficiente C': f"{analysis['data_preview']['C_value']:.4f}",
                'UVA-PF': f"{analysis['data_preview'].get('uva_pf', 0):.2f}" if analysis['data_preview'].get('uva_pf') else 'N/A'
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Gráfico comparativo
        if len(st.session_state.analysis_history) > 1:
            st.subheader("📈 Comparativo Gráfico")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            analyses = [a['name'] for a in st.session_state.analysis_history]
            spf_values = [a['data_preview']['spf'] for a in st.session_state.analysis_history]
            adjusted_values = [a['data_preview']['spf_adjusted'] for a in st.session_state.analysis_history]
            
            x = np.arange(len(analyses))
            width = 0.35
            
            ax.bar(x - width/2, spf_values, width, label='SPF In Vitro', alpha=0.8, color='blue')
            ax.bar(x + width/2, adjusted_values, width, label='SPF Ajustado', alpha=0.8, color='orange')
            
            ax.set_xlabel('Análises')
            ax.set_ylabel('Valor do SPF')
            ax.set_title('Comparação entre Análises')
            ax.set_xticks(x)
            ax.set_xticklabels(analyses, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("📝 Nenhuma análise salva no histórico. Execute análises para comparar resultados.")

elif page == "Explicação das Equações":
    st.header("📚 Explicação das Equações Matemáticas")
    
    tab1, tab2, tab3 = st.tabs(["SPF", "UVA-PF", "Métricas Avançadas"])
    
    with tab1:
        st.subheader("Cálculo do Fator de Proteção Solar (SPF)")
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
        
        st.subheader("SPF Ajustado com Coeficiente C")
        st.latex(r'''
        SPF_{\text{ajustado}} = \frac{\sum E(\lambda) \times I(\lambda) \times \Delta\lambda}
        {\sum E(\lambda) \times I(\lambda) \times 10^{-A(\lambda) \times C} \times \Delta\lambda}
        ''')
        st.markdown("""
        **Onde:**
        - $C$ = Coeficiente de ajuste que correlaciona o SPF in vitro com o SPF in vivo
        """)
    
    with tab2:
        st.subheader("Cálculo do UVA Protection Factor (UVA-PF)")
        st.latex(r'''
        UVA\text{-}PF = \frac{\sum P(\lambda) \times I(\lambda) \times \Delta\lambda}
        {\sum P(\lambda) \times I(\lambda) \times 10^{-A_e(\lambda) \times C} \times \Delta\lambda}
        ''')
        st.markdown("""
        **Onde:**
        - $P(\lambda)$ = Espectro de pigmentação UVA
        - $I(\lambda)$ = Intensidade spectral
        - $A_e(\lambda)$ = Absorbância após irradiação
        - $C$ = Coeficiente de ajuste
        - $\Delta\lambda$ = Intervalo entre comprimentos de onda (1 nm)
        """)
        
        st.markdown("""
        **Critérios de Avaliação UVA:**
        - **UVA-PF/SPF ≥ 1/3**: Boa proteção UVA relativa
        - **λ Crítico ≥ 370 nm**: Amplo espectro de proteção
        """)
    
    with tab3:
        st.subheader("Comprimento de Onda Crítico (λc)")
        st.latex(r'''
        \lambda_c = \min \left\{ \lambda \middle| \int_{290}^{\lambda} A(\lambda)  d\lambda \geq 0.9 \times \int_{290}^{400} A(\lambda)  d\lambda \right\}
        ''')
        st.markdown("""
        **Interpretação:**
        - λc ≥ 370 nm: Proteção UVA adequada (requisito regulatório)
        - λc < 370 nm: Proteção UVA potencialmente insuficiente
        """)
        
        st.subheader("Razão UVA/UV")
        st.latex(r'''
        \text{Razão} = \frac{\int_{320}^{400} A(\lambda)  d\lambda}
        {\int_{290}^{400} A(\lambda)  d\lambda}
        ''')
        st.markdown("""
        **Interpretação:**
        - Mede a proporção da proteção UVA em relação à proteção UV total
        - Valores mais altos indicam melhor proteção UVA relativa
        """)

# Rodapé
st.markdown("---")
st.markdown("""
**🔬 Sistema de Análise de Proteção Solar**  
*Desenvolvido para avaliação completa de fotoproteção*
""")

# Instruções de uso
with st.sidebar:
    st.markdown("---")
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
        1. **Cálculo SPF**: 
           - Carregue dados pré-irradiação
           - Calcule SPF in vitro
           - Ajuste com SPF rotulado para obter coeficiente C
        
        2. **Análise UVA**: 
           - Carregue dados pós-irradiação (wavelength, P, I, A_e)
           - Use o coeficiente C calculado
           - Calcule UVA-PF e métricas
        
        3. **Verifique conformidade** com os critérios regulatórios
        """)
