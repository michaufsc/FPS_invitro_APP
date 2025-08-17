import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from PIL import Image
import os
from pathlib import Path

# ===================== CONFIGURAÇÃO INICIAL =====================
st.set_page_config(
    layout="wide",
    page_title="Fotoproteção In Vitro",
    page_icon="🌞"
)

# ===================== CONSTANTES E REFERÊNCIAS =====================
'''
REFERÊNCIAS CIENTÍFICAS:
1. ISO 24443:2012 - In vitro determination of sunscreen UVA protection
2. Mansur et al. (1986) - Correlação in vitro/in vivo para SPF
3. COLIPA (2009) - Guideline for UVA protection testing
4. HPC Today (2024) - Métodos para UVA1 e Ultra-Longo UVA
'''

# ===================== FUNÇÕES AUXILIARES =====================
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

def validar_dados(df):
    """Valida a estrutura do DataFrame"""
    colunas_necessarias = {
        'Comprimento de Onda': 'nm',
        'Absorbancia': 'adimensional',
        'E(λ)': 'Eritematosa (CIE 1998)',
        'I(λ)': 'Irradiância solar (ISO/COLIPA)',
        'P(λ)': 'Espectro UVA (COLIPA)'
    }
    
    for col, desc in colunas_necessarias.items():
        if col not in df.columns:
            st.error(f"Coluna obrigatória não encontrada: {col} ({desc})")
            return False
    
    if df['Comprimento de Onda'].min() > 290 or df['Comprimento de Onda'].max() < 400:
        st.warning("Faixa espectral incompleta (290-400 nm recomendado)")
    
    return True

# ===================== INTERFACE PRINCIPAL =====================
# Cabeçalho com fallback responsivo
col1, col2 = st.columns([1, 0.2])
with col1:
    logo = carregar_imagem("logo.png", 200)
    if logo:
        st.image(logo)
    else:
        st.title("🌞 Fotoproteção In Vitro")

with col2:
    logo_parceiro = carregar_imagem("logo_parceiro.png", 100)
    if logo_parceiro:
        st.image(logo_parceiro)

# ===================== ABAS PRINCIPAIS =====================
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

    **Ajuste para SPF in vivo:**
    $$
    SPF_{ajustado} = \frac{\int E \cdot I \, d\lambda}{\int E \cdot I \cdot 10^{-A \cdot C} \, d\lambda}
    $$
    """)
    
    # Upload de dados
    arquivo = st.file_uploader("Carregue arquivo de espectros", type=["csv", "xlsx"])
    
    if arquivo:
        try:
            # Leitura dos dados
            df = pd.read_csv(arquivo) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo)
            df.columns = df.columns.str.strip()
            
            if not validar_dados(df):
                st.stop()
            
            # Pré-processamento
            df['Transmitancia'] = 10 ** (-df['Absorbancia'])
            delta_lambda = np.mean(np.diff(df['Comprimento de Onda']))
            
            # Cálculo do SPF
            def calcular_spf(df, C=1.0):
                numerador = np.trapz(df['E(λ)'] * df['I(λ)'], x=df['Comprimento de Onda'])
                denominador = np.trapz(df['E(λ)'] * df['I(λ)'] * 10**(-df['Absorbancia'] * C), 
                                    x=df['Comprimento de Onda'])
                return numerador / denominador
            
            spf = calcular_spf(df)
            st.success(f"**SPF In Vitro:** {spf:.2f}")
            
            # Ajuste do coeficiente C
            st.subheader("🔧 Ajuste para SPF Rotulado")
            spf_alvo = st.number_input("SPF Rotulado In Vivo", min_value=1.0, value=30.0, step=0.5)
            
            def erro(C):
                return abs(calcular_spf(df, C) - spf_alvo)
            
            resultado = opt.minimize_scalar(erro, bounds=(0.5, 1.5), method='bounded')
            C_ajustado = resultado.x
            st.session_state.C_ajustado = C_ajustado
            st.session_state.df_iso = df
            
            st.success(f"""
            **Resultados do Ajuste:**
            - Coeficiente C: {C_ajustado:.4f}
            - SPF Ajustado: {calcular_spf(df, C_ajustado):.2f}
            """)
            
            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['Comprimento de Onda'], df['Absorbancia'], label='Absorbância')
            ax.set_xlabel("Comprimento de Onda (nm)")
            ax.set_ylabel("Absorbância")
            ax.grid(True)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro no processamento: {str(e)}")

# ===================== ABA 2: MÉTODO MANSUR =====================
with tab2:
    st.header("⚡ Método de Mansur (1986) - SPF UVB")
    
    st.markdown(r"""
    **Fórmula simplificada (290-320 nm):**
    $$
    SPF = 10 \times \sum_{\lambda=290}^{320} EE(\lambda) \cdot I(\lambda) \cdot A(\lambda)
    $$
    """)
    
    if 'df_iso' in st.session_state:
        df = st.session_state.df_iso
        df_mansur = df[(df['Comprimento de Onda'] >= 290) & (df['Comprimento de Onda'] <= 320)].copy()
        
        # Tabela de coeficientes EE*I (Mansur 1986)
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
    else:
        st.warning("Carregue os dados na aba ISO 24443 primeiro")

# ===================== ABA 3: CWC =====================
with tab3:
    st.header("📏 Comprimento de Onda Crítico (CWC)")
    
    st.markdown(r"""
    **Definição (ISO 24443):**
    $$
    CWC = \lambda \ \text{onde} \ \frac{\int_{290}^{\lambda} A(\lambda) d\lambda}{\int_{290}^{400} A(\lambda) d\lambda} \geq 0.9
    $$
    """)
    
    if 'df_iso' in st.session_state:
        df = st.session_state.df_iso
        df_cwc = df[(df['Comprimento de Onda'] >= 290) & (df['Comprimento de Onda'] <= 400)].copy()
        
        # Cálculo das áreas
        area_total = np.trapz(df_cwc['Absorbancia'], x=df_cwc['Comprimento de Onda'])
        area_cumulativa = np.cumsum(df_cwc['Absorbancia'] * np.gradient(df_cwc['Comprimento de Onda']))
        
        # Encontrar CWC
        idx_cwc = np.where(area_cumulativa >= 0.9 * area_total)[0][0]
        cwc = df_cwc['Comprimento de Onda'].iloc[idx_cwc]
        
        st.success(f"""
        **Resultado:**
        - CWC = {cwc:.1f} nm
        - {'✅ Proteção UVA ampla (≥370 nm)' if cwc >= 370 else '⚠️ Proteção UVA insuficiente'}
        """)
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_cwc['Comprimento de Onda'], df_cwc['Absorbancia'], label='Absorbância')
        ax.axvline(cwc, color='red', linestyle='--', label=f'CWC = {cwc:.1f} nm')
        ax.fill_between(df_cwc['Comprimento de Onda'], 0, df_cwc['Absorbancia'], alpha=0.2)
        ax.set_xlabel("Comprimento de Onda (nm)")
        ax.set_ylabel("Absorbância")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Carregue os dados na aba ISO 24443 primeiro")

# ===================== ABA 4: UVA-PF =====================
with tab4:
    st.header("🟣 Fator de Proteção UVA (UVA-PF)")
    
    st.markdown(r"""
    **Fórmula (ISO 24443):**
    $$
    UVA\!-\!PF = \frac{\int_{320}^{400} P(\lambda) \cdot I(\lambda) d\lambda}{\int_{320}^{400} P(\lambda) \cdot I(\lambda) \cdot 10^{-A(\lambda) \cdot C} d\lambda}
    $$
    """)
    
    if 'df_iso' in st.session_state and 'C_ajustado' in st.session_state:
        df = st.session_state.df_iso
        C = st.session_state.C_ajustado
        
        df_uva = df[(df['Comprimento de Onda'] >= 320) & (df['Comprimento de Onda'] <= 400)].copy()
        
        numerador = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'], x=df_uva['Comprimento de Onda'])
        denominador = np.trapz(df_uva['P(λ)'] * df_uva['I(λ)'] * 10**(-df_uva['Absorbancia'] * C), 
                         x=df_uva['Comprimento de Onda'])
        uva_pf = numerador / denominador
        
        st.success(f"**UVA-PF:** {uva_pf:.2f}")
        
        # Verificação da relação UVA-PF/SPF
        if 'spf_iso' in st.session_state:
            relacao = uva_pf / st.session_state.spf_iso
            st.info(f"**Relação UVA-PF/SPF:** {relacao:.2f} (Requisito: ≥0.33)")
    else:
        st.warning("Execute o ajuste na aba ISO 24443 primeiro")

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
    
    if 'df_iso' in st.session_state and 'C_ajustado' in st.session_state:
        df = st.session_state.df_iso
        C = st.session_state.C_ajustado
        
        # Cálculo UVA1-PF
        df_uva1 = df[(df['Comprimento de Onda'] >= 340) & (df['Comprimento de Onda'] <= 400)]
        num_uva1 = np.trapz(df_uva1['E(λ)'] * df_uva1['I(λ)'], x=df_uva1['Comprimento de Onda'])
        den_uva1 = np.trapz(df_uva1['E(λ)'] * df_uva1['I(λ)'] * 10**(-df_uva1['Absorbancia'] * C), 
                          x=df_uva1['Comprimento de Onda'])
        uva1_pf = num_uva1 / den_uva1
        
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
        
        # Gráfico comparativo
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['Comprimento de Onda'], df['Absorbancia'], label='Absorbância Total')
        ax.axvspan(340, 400, color='purple', alpha=0.1, label='UVA1 (340-400 nm)')
        ax.axvspan(370, 400, color='red', alpha=0.1, label='Ultra-Longo UVA (370-400 nm)')
        ax.set_xlabel("Comprimento de Onda (nm)")
        ax.set_ylabel("Absorbância")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Execute o ajuste na aba ISO 24443 primeiro")

# ===================== RODAPÉ =====================
st.markdown("---")
st.caption("""
**Referências Científicas:**  
1. ISO 24443:2012 - Determination of sunscreen UVA photoprotection in vitro  
2. Mansur, J.S. et al. (1986) - Determinação do Fator de Proteção Solar por Espectrofotometria  
3. COLIPA (2009) - International sun protection factor (SPF) test method  
4. HPC Today (2024) - In vitro method for UVA1, long UVA or ultra-long UVA claiming  
""")

# ===================== REQUIREMENTS.TXT =====================
'''
streamlit==1.32.2
pandas==2.1.4
numpy==1.26.3
matplotlib==3.8.2
scipy==1.11.4
Pillow==10.1.0
openpyxl==3.1.2
'''
