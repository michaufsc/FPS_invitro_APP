import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from PIL import Image
import os
from pathlib import Path
import plotly.express as px
import io

# ===================== CONFIGURAÇÃO INICIAL =====================
st.set_page_config(
    layout="wide",
    page_title="Fotoproteção In Vitro",
    page_icon="🌞"
)

# ===================== CONSTANTES E REFERÊNCIAS =====================
REFERENCIAS = """
**Referências Científicas:**  
1. ISO 24443:2012 - Determinação in vitro da proteção UVA de filtros solares  
2. Mansur et al. (1986) - Correlação in vitro/in vivo para SPF  
3. COLIPA (2009) - Guia para testes de proteção UVA  
4. HPC Today (2024) - Métodos para UVA1 e Ultra-Longo UVA  
"""

# ===================== FUNÇÕES AUXILIARES =====================
@st.cache_data
def carregar_imagem(nome_arquivo, largura):
    """Carrega imagens com busca em múltiplos diretórios"""
    try:
        diretorios = ["", "images/", "assets/", "static/"]
        for dir in diretorios:
            caminho = Path(dir) / nome_arquivo
            if caminho.exists():
                img = Image.open(caminho)
                return img.resize((largura, int(largura * img.size[1]/img.size[0])))
        return None
    except Exception as e:
        st.error(f"Erro ao carregar imagem: {str(e)}")
        return None

@st.cache_data
def carregar_dados(arquivo):
    """Carrega dados com cache para melhor performance"""
    try:
        if arquivo.name.endswith('.csv'):
            return pd.read_csv(arquivo)
        return pd.read_excel(arquivo)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def validar_dados(df):
    """Validação completa da estrutura e qualidade dos dados"""
    colunas_necessarias = {
        'Comprimento de Onda': {'unidade': 'nm', 'faixa': (290, 400)},
        'Absorbancia': {'unidade': 'UA', 'faixa': (0, 3)},
        'E(λ)': {'unidade': 'normalizado', 'faixa': (0, 1)},
        'I(λ)': {'unidade': 'normalizado', 'faixa': (0, 1)},
        'P(λ)': {'unidade': 'normalizado', 'faixa': (0, 1)}
    }
    
    erros = []
    for coluna, especs in colunas_necessarias.items():
        if coluna not in df.columns:
            erros.append(f"Coluna obrigatória faltante: {coluna}")
            continue
            
        dados_coluna = df[coluna]
        if not np.issubdtype(dados_coluna.dtype, np.number):
            erros.append(f"Coluna {coluna} contém valores não numéricos")
        
        if 'faixa' in especs:
            if (dados_coluna.min() < especs['faixa'][0]) or (dados_coluna.max() > especs['faixa'][1]):
                erros.append(f"Coluna {coluna} fora da faixa esperada {especs['faixa']}")
    
    if erros:
        st.error("Erros na validação dos dados:")
        for erro in erros:
            st.error(f"• {erro}")
        return False
    
    passo_espectral = np.mean(np.diff(df['Comprimento de Onda']))
    if passo_espectral > 5:
        st.warning(f"Intervalo espectral grande ({passo_espectral:.1f} nm). Recomenda-se ≤5nm para precisão")
    
    return True

def plotar_espectro(df, titulo="Espectro de Absorbância"):
    """Cria gráfico interativo com Plotly"""
    fig = px.line(df, x='Comprimento de Onda', y='Absorbancia', 
                 title=titulo, labels={'Absorbancia': 'Absorbância (UA)'})
    
    fig.add_vrect(x0=290, x1=320, fillcolor="blue", opacity=0.1, 
                 annotation_text="UVB", line_width=0)
    fig.add_vrect(x0=320, x1=400, fillcolor="purple", opacity=0.1, 
                 annotation_text="UVA", line_width=0)
    
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Comprimento de Onda (nm)",
        yaxis_title="Absorbância",
        template="plotly_white"
    )
    return fig

def calcular_spf(df, C=1.0, lambda_min=290, lambda_max=400):
    """Calcula SPF para faixa espectral definida"""
    df_faixa = df[(df['Comprimento de Onda'] >= lambda_min) & 
                (df['Comprimento de Onda'] <= lambda_max)]
    
    numerador = np.trapz(df_faixa['E(λ)'] * df_faixa['I(λ)'], 
                        x=df_faixa['Comprimento de Onda'])
    denominador = np.trapz(df_faixa['E(λ)'] * df_faixa['I(λ)'] * 
                         10**(-df_faixa['Absorbancia'] * C), 
                        x=df_faixa['Comprimento de Onda'])
    return numerador / denominador

def calcular_uva_pf(df, C=1.0):
    """Calcula UVA-PF conforme ISO 24443"""
    df_uva = df[(df['Comprimento de Onda'] >= 320) & 
               (df['Comprimento de Onda'] <= 400)]
    
    numerador = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'], 
                        x=df_uva['Comprimento de Onda'])
    denominador = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'] * 
                         10**(-df_uva['Absorbancia'] * C), 
                        x=df_uva['Comprimento de Onda'])
    return numerador / denominador

def calcular_cwc(df):
    """Calcula Comprimento de Onda Crítico (CWC)"""
    df_faixa = df[(df['Comprimento de Onda'] >= 290) & 
                (df['Comprimento de Onda'] <= 400)]
    
    area_total = np.trapz(df_faixa['Absorbancia'], 
                         x=df_faixa['Comprimento de Onda'])
    area_acumulada = np.cumsum(df_faixa['Absorbancia'] * 
                              np.gradient(df_faixa['Comprimento de Onda']))
    
    idx_cwc = np.where(area_acumulada >= 0.9 * area_total)[0][0]
    return df_faixa['Comprimento de Onda'].iloc[idx_cwc]

def download_excel(df_resultados):
    """Gera arquivo Excel para download"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_resultados.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

# ===================== INTERFACE PRINCIPAL =====================
# Cabeçalho
col1, col2 = st.columns([1, 0.2])
with col1:
    logo = carregar_imagem("logo.png", 200)
    if logo:
        st.image(logo)
    else:
        st.title("🌞 Fotoproteção In Vitro")

with col2:
    logo_parceiro = carregar_imagem("logo_ufsc.png", 100)
    if logo_parceiro:
        st.image(logo_parceiro)

# Menu de navegação
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ISO 24443 (SPF)", 
    "Mansur (SPF UVB)", 
    "CWC", 
    "UVA-PF", 
    "UVA1/Ultra-Longo"
])

# ===================== ABA 1: ISO 24443 =====================
with tab1:
    st.header("📊 Método ISO 24443 - SPF In Vitro")
    
    st.markdown(r"""
    **Fórmula do SPF:**
    $$
    SPF = \frac{\int_{290}^{400} E(\lambda) \cdot I(\lambda) \, d\lambda}{\int_{290}^{400} E(\lambda) \cdot I(\lambda) \cdot 10^{-A(\lambda)} \, d\lambda}
    $$
    """)
    
    # Upload de dados
    arquivo = st.file_uploader("Carregue arquivo de espectros", type=["csv", "xlsx"])
    
    if arquivo:
        df = carregar_dados(arquivo)
        if df is not None:
            df.columns = df.columns.str.strip()
            
            if validar_dados(df):
                # Pré-processamento
                df['Transmitancia'] = 10 ** (-df['Absorbancia'])
                
                # Cálculo inicial
                spf_inicial = calcular_spf(df)
                st.success(f"**SPF In Vitro:** {spf_inicial:.2f}")
                
                # Ajuste interativo
                st.subheader("🔧 Ajuste para SPF Rotulado")
                spf_alvo = st.number_input("SPF Rotulado In Vivo", 
                                         min_value=1.0, value=30.0, step=0.5)
                
                def erro(C):
                    return abs(calcular_spf(df, C) - spf_alvo)
                
                resultado = opt.minimize_scalar(erro, bounds=(0.5, 1.5), method='bounded')
                C_ajustado = resultado.x
                st.session_state.C_ajustado = C_ajustado
                st.session_state.df = df
                
                st.success(f"""
                **Resultados do Ajuste:**
                - Coeficiente C: {C_ajustado:.4f}
                - SPF Ajustado: {calcular_spf(df, C_ajustado):.2f}
                """)
                
                # Gráfico
                st.plotly_chart(plotar_espectro(df), use_container_width=True)

# ===================== ABA 2: MÉTODO MANSUR =====================
with tab2:
    st.header("⚡ Método de Mansur (1986) - SPF UVB")
    
    st.markdown(r"""
    **Fórmula simplificada (290-320 nm):**
    $$
    SPF = 10 \times \sum_{\lambda=290}^{320} EE(\lambda) \cdot I(\lambda) \cdot A(\lambda)
    $$
    """)
    
    if 'df' in st.session_state:
        df = st.session_state.df
        df_mansur = df[(df['Comprimento de Onda'] >= 290) & 
                      (df['Comprimento de Onda'] <= 320)].copy()
        
        # Coeficientes EE*I (Mansur 1986)
        coeficientes = {
            290: 0.0150 * 0.134, 295: 0.0817 * 0.134,
            300: 0.2874 * 0.135, 305: 0.3278 * 0.136,
            310: 0.1864 * 0.137, 315: 0.0839 * 0.138,
            320: 0.0180 * 0.139
        }
        
        df_mansur['EE_I'] = df_mansur['Comprimento de Onda'].map(coeficientes)
        df_mansur['Contribuição'] = df_mansur['EE_I'] * df_mansur['Absorbancia']
        spf_mansur = 10 * df_mansur['Contribuição'].sum()
        
        st.success(f"**SPF (Mansur):** {spf_mansur:.2f}")
        
        # Tabela detalhada
        with st.expander("🔍 Ver detalhes do cálculo"):
            st.dataframe(df_mansur[['Comprimento de Onda', 'Absorbancia', 'EE_I', 'Contribuição']])

# ===================== ABA 3: CWC =====================
with tab3:
    st.header("📏 Comprimento de Onda Crítico (CWC)")
    
    st.markdown(r"""
    **Definição (ISO 24443):**
    $$
    CWC = \lambda \ \text{onde} \ \frac{\int_{290}^{\lambda} A(\lambda) d\lambda}{\int_{290}^{400} A(\lambda) d\lambda} \geq 0.9
    $$
    """)
    
    if 'df' in st.session_state:
        df = st.session_state.df
        cwc = calcular_cwc(df)
        
        st.success(f"""
        **Resultado:**
        - CWC = {cwc:.1f} nm
        - {'✅ Proteção UVA ampla (≥370 nm)' if cwc >= 370 else '⚠️ Proteção UVA insuficiente'}
        """)
        
        # Gráfico
        fig = px.line(df, x='Comprimento de Onda', y='Absorbancia')
        fig.add_vline(x=cwc, line_dash="dash", line_color="red",
                     annotation_text=f"CWC = {cwc:.1f} nm")
        fig.update_layout(title="Determinação do CWC")
        st.plotly_chart(fig, use_container_width=True)

# ===================== ABA 4: UVA-PF =====================
with tab4:
    st.header("🟣 Fator de Proteção UVA (UVA-PF)")
    
    st.markdown(r"""
    **Fórmula (ISO 24443):**
    $$
    UVA\!-\!PF = \frac{\int_{320}^{400} P(\lambda) \cdot I(\lambda) d\lambda}{\int_{320}^{400} P(\lambda) \cdot I(\lambda) \cdot 10^{-A(\lambda) \cdot C} d\lambda}
    $$
    """)
    
    if 'df' in st.session_state and 'C_ajustado' in st.session_state:
        df = st.session_state.df
        C = st.session_state.C_ajustado
        
        uva_pf = calcular_uva_pf(df, C)
        st.success(f"**UVA-PF:** {uva_pf:.2f}")
        
        # Verificação da relação UVA-PF/SPF
        spf_ajustado = calcular_spf(df, C)
        relacao = uva_pf / spf_ajustado
        st.info(f"**Relação UVA-PF/SPF:** {relacao:.2f} (Requisito: ≥0.33)")

# ===================== ABA 5: UVA1 E ULTRA-LONGO =====================
with tab5:
    st.header("🔮 Métricas UVA1 e Ultra-Longo UVA")
    
    st.markdown(r"""
    **Baseado em HPC Today (2024):**
    - **UVA1-PF (340-400 nm):** 
    $$
    \frac{\int_{340}^{400} E(\lambda)I(\lambda)d\lambda}{\int_{340}^{400} E(\lambda)I(\lambda)10^{-A(\lambda)C}d\lambda}
    $$
    - **Absorbância média (370-400 nm):**
    $$
    \frac{1}{N}\sum_{\lambda=370}^{400} A(\lambda)
    $$
    """)
    
    if 'df' in st.session_state and 'C_ajustado' in st.session_state:
        df = st.session_state.df
        C = st.session_state.C_ajustado
        
        # Cálculo UVA1-PF
        uva1_pf = calcular_spf(df, C, 340, 400)
        
        # Cálculo absorbância média
        df_ultra = df[df['Comprimento de Onda'] >= 370]
        media_abs = df_ultra['Absorbancia'].mean()
        
        # Exibição dos resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric("UVA1-PF (340-400 nm)", f"{uva1_pf:.2f}")
            
        with col2:
            st.metric("Absorbância Média (370-400 nm)", 
                    f"{media_abs:.3f}",
                    "✅ ≥ 0.8 (Boa proteção)" if media_abs >= 0.8 else "⚠️ < 0.8")

# ===================== RODAPÉ E DOWNLOADS =====================
st.markdown("---")

if 'df' in st.session_state:
    st.subheader("📤 Exportar Resultados")
    
    # Preparar dados para exportação
    resultados = {
        "Parâmetro": ["SPF In Vitro", "SPF Ajustado", "Coeficiente C", 
                     "SPF Mansur", "UVA-PF", "CWC", "UVA1-PF", 
                     "Absorbância Média (370-400nm)"],
        "Valor": [
            calcular_spf(st.session_state.df),
            calcular_spf(st.session_state.df, st.session_state.get('C_ajustado', 1.0)),
            st.session_state.get('C_ajustado', 1.0),
            # Outros valores calculados...
        ]
    }
    df_resultados = pd.DataFrame(resultados)
    
    # Botão de download
    excel_data = download_excel(df_resultados)
    st.download_button(
        label="📥 Baixar Resultados em Excel",
        data=excel_data,
        file_name="resultados_fotoprotecao.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Referências
st.markdown(REFERENCIAS)
