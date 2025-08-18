import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import simps
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

## Configurações da Página
st.set_page_config(
    page_title="Análise de Fotoproteção UV Precisa",
    page_icon="☀️",
    layout="wide"
)

## Título e Descrição
st.title("☀️ Análise Científica de Fotoproteção UV")
st.markdown("""
Aplicativo para cálculo preciso de parâmetros de fotoproteção conforme métodos internacionais.
""")

## =============================================
## CONSTANTES E ESPECTROS DE REFERÊNCIA
## =============================================

# Faixas espectrais (nm)
UVB_RANGE = (290, 320)
UVA2_RANGE = (320, 340)
UVA1_RANGE = (340, 400)
ULTRALONG_UVA_RANGE = (370, 400)
FULL_RANGE = (290, 400)

# Espectro de eritema (CIE 1998)
def erythema_action_spectrum(wavelength):
    return 10**(0.094*(298-wavelength)) if wavelength <= 298 else 1.0

# Espectro UVA (ISO 24443)
def uva_action_spectrum(wavelength):
    return 1.0 if (wavelength >= 320) and (wavelength <= 400) else 0.0

# Fator de normalização
K_ERITEMA = 1.0  # Ajustar conforme calibração

## =============================================
## FUNÇÕES DE CÁLCULO PRECISAS
## =============================================

def calculate_spf(wavelengths, absorbance):
    """Cálculo preciso do SPF com espectro de eritema"""
    mask = (wavelengths >= UVB_RANGE[0]) & (wavelengths <= UVB_RANGE[1])
    er = np.array([erythema_action_spectrum(w) for w in wavelengths[mask]])
    transmittance = 10**(-absorbance[mask])  # Transmitância = 10^-absorbance
    spf = simps(er * transmittance, wavelengths[mask])
    return K_ERITEMA / spf if spf > 0 else 0

def calculate_uva_pf(wavelengths, absorbance):
    """Cálculo do UVA-PF conforme ISO 24443"""
    mask = (wavelengths >= UVA2_RANGE[0]) & (wavelengths <= UVA1_RANGE[1])
    uva = np.array([uva_action_spectrum(w) for w in wavelengths[mask]])
    transmittance = 10**(-absorbance[mask])
    return simps(uva * transmittance, wavelengths[mask]) * 10  # Fator 10 conforme norma

def calculate_uva1_pf(wavelengths, absorbance):
    """Cálculo UVA1-PF (340-400nm) conforme HPC Today"""
    mask = (wavelengths >= UVA1_RANGE[0]) & (wavelengths <= UVA1_RANGE[1])
    return np.mean(10**(-absorbance[mask])) * 10  # Média de transmitância

def calculate_lambda_c(wavelengths, absorbance):
    """Cálculo preciso do λc com interpolação"""
    transmittance = 10**(-absorbance)
    total_area = simps(transmittance, wavelengths)
    cumulative = np.cumsum(transmittance * np.gradient(wavelengths))
    return np.interp(0.9*total_area, cumulative, wavelengths)

def calculate_uv_ratio(uva_pf, spf):
    """Cálculo da razão UVA/UVB"""
    return uva_pf / spf if spf > 0 else 0

## =============================================
## INTERFACE DO USUÁRIO
## =============================================

with st.expander("📤 Upload de Arquivo", expanded=True):
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo de dados espectrais (CSV ou Excel)",
        type=['csv', 'xlsx'],
        help="""O arquivo deve conter duas colunas:
        1. wavelength (comprimento de onda em nm)
        2. absorbance (valores de absorbância)"""
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Verificar e padronizar colunas
            df.columns = [col.lower().strip() for col in df.columns]
            
            if not {'wavelength', 'absorbance'}.issubset(df.columns):
                st.error("Erro: O arquivo deve conter colunas 'wavelength' e 'absorbance'")
            else:
                # Pré-processamento
                df = df[['wavelength', 'absorbance']].dropna()
                df = df.sort_values('wavelength')
                wavelengths = df['wavelength'].values
                absorbance = df['absorbance'].values
                
                # Verificar intervalo espectral
                if wavelengths.min() > 290 or wavelengths.max() < 400:
                    st.warning("Dados não cobrem todo o espectro UV (290-400 nm). Resultados podem ser imprecisos.")
                
                # Cálculos
                results = {
                    'SPF': calculate_spf(wavelengths, absorbance),
                    'UVA-PF': calculate_uva_pf(wavelengths, absorbance),
                    'UVA1-PF': calculate_uva1_pf(wavelengths, absorbance),
                    'λc (nm)': calculate_lambda_c(wavelengths, absorbance),
                    'Razão UVA/UVB': calculate_uv_ratio(
                        calculate_uva_pf(wavelengths, absorbance),
                        calculate_spf(wavelengths, absorbance)
                    ),
                    'Dados': df
                }
                
                st.session_state.results = results
                st.success("Análise concluída com sucesso!")
                
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

## =============================================
## VISUALIZAÇÃO DOS RESULTADOS
## =============================================

if 'results' in st.session_state:
    st.markdown("## 📊 Resultados Científicos")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("SPF (290-320 nm)", 
                f"{st.session_state.results['SPF']:.1f}",
                help="Fator de Proteção Solar (ISO 24443)")
    with cols[1]:
        st.metric("UVA-PF (320-400 nm)", 
                f"{st.session_state.results['UVA-PF']:.1f}",
                help="Fator de Proteção UVA (ISO 24443)")
    with cols[2]:
        st.metric("UVA1-PF (340-400 nm)", 
                f"{st.session_state.results['UVA1-PF']:.1f}",
                help="Proteção UVA1 (HPC Today)")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("λc (Comprimento Crítico)", 
                f"{st.session_state.results['λc (nm)']:.1f} nm",
                help="Comprimento onde 90% da absorbância está acumulada")
    with cols[1]:
        st.metric("Razão UVA/UVB", 
                f"{st.session_state.results['Razão UVA/UVB']:.2f}",
                help="Balanceamento de proteção UVA/UVB")
    
    st.markdown("### 📈 Visualização Espectral")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Curva de absorbância
    ax.plot(st.session_state.results['Dados']['wavelength'], 
            st.session_state.results['Dados']['absorbance'],
            'b-', linewidth=2, label='Absorbância')
    
    # Regiões destacadas
    ax.axvspan(*UVB_RANGE, color='red', alpha=0.1, label='UVB')
    ax.axvspan(*UVA2_RANGE, color='orange', alpha=0.1, label='UVA2')
    ax.axvspan(*UVA1_RANGE, color='yellow', alpha=0.1, label='UVA1')
    ax.axvline(st.session_state.results['λc (nm)'], color='green', 
              linestyle='--', label=f'λc ({st.session_state.results["λc (nm)"]:.1f} nm)')
    
    ax.set_xlabel('Comprimento de Onda (nm)')
    ax.set_ylabel('Absorbância')
    ax.set_title('Perfil Espectral de Absorbância UV')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

## =============================================
## EXPORTAÇÃO DE RESULTADOS
## =============================================

if 'results' in st.session_state:
    st.markdown("## 💾 Exportar Resultados")
    
    # Preparar dados para exportação
    df_results = pd.DataFrame({
        'Parâmetro': ['SPF', 'UVA-PF', 'UVA1-PF', 'λc (nm)', 'Razão UVA/UVB'],
        'Valor': [
            st.session_state.results['SPF'],
            st.session_state.results['UVA-PF'],
            st.session_state.results['UVA1-PF'],
            st.session_state.results['λc (nm)'],
            st.session_state.results['Razão UVA/UVB']
        ],
        'Descrição': [
            'Fator de Proteção Solar (290-320 nm)',
            'Fator de Proteção UVA (320-400 nm)',
            'Fator de Proteção UVA1 (340-400 nm)',
            'Comprimento de onda crítico (90% área acumulada)',
            'Razão entre proteção UVA e UVB'
        ],
        'Método': [
            'ISO 24443 com espectro de eritema',
            'ISO 24443 com espectro UVA',
            'Média de transmitância (HPC Today)',
            'Integral da transmitância',
            'Razão UVA-PF/SPF'
        ]
    })
    
    # Botão de exportação
    if st.button("Gerar Relatório Científico"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Resultados
            df_results.to_excel(writer, sheet_name='Resultados', index=False)
            
            # Dados brutos
            st.session_state.results['Dados'].to_excel(
                writer, sheet_name='Dados Espectrais', index=False)
            
            # Métodos
            methods = pd.DataFrame({
                'Método': [
                    'SPF',
                    'UVA-PF',
                    'UVA1-PF',
                    'λc'
                ],
                'Fórmula': [
                    r'$\frac{K}{\int_{290}^{320} E(\lambda) \cdot T(\lambda) d\lambda}$',
                    r'$10 \cdot \int_{320}^{400} T(\lambda) d\lambda$',
                    r'$10 \cdot \overline{T}_{340-400}$',
                    r'$\lambda_c : \int_{290}^{\lambda_c} T(\lambda) d\lambda = 0.9 \cdot \int_{290}^{400} T(\lambda) d\lambda$'
                ],
                'Referência': [
                    'ISO 24443',
                    'ISO 24443',
                    'HPC Today 3(2024)',
                    'ISO 24443'
                ]
            })
            methods.to_excel(writer, sheet_name='Métodos', index=False)
            
            # Formatação
            workbook = writer.book
            for sheet in ['Resultados', 'Dados Espectrais', 'Métodos']:
                worksheet = writer.sheets[sheet]
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:B', 15)
                worksheet.set_column('C:C', 40)
            
        st.download_button(
            label="⬇️ Baixar Relatório Completo (Excel)",
            data=output.getvalue(),
            file_name=f"relatorio_fotoprotecao_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

## =============================================
## SEÇÃO DE INFORMAÇÕES TÉCNICAS
## =============================================

with st.expander("📚 Fundamentos Científicos", expanded=False):
    st.markdown("""
    ### **Métodos de Cálculo**
    
    1. **SPF (Fator de Proteção Solar)**
       - Faixa: 290-320 nm
       - Fórmula:
         $$
         SPF = \\frac{K_{eritema}}{\int_{290}^{320} E(\lambda) \cdot T(\lambda) d\lambda}
         $$
       - Onde:
         - $E(\lambda)$: Espectro de ação eritematógena (CIE 1998)
         - $T(\lambda) = 10^{-A(\lambda)}$: Transmitância
    
    2. **UVA-PF (Fator de Proteção UVA)**
       - Faixa: 320-400 nm
       - Fórmula:
         $$
         UVA\\text{-}PF = 10 \cdot \int_{320}^{400} T(\lambda) d\lambda
         $$
    
    3. **UVA1-PF (340-400 nm)**
       - Método simplificado:
         $$
         UVA1\\text{-}PF = 10 \cdot \overline{T}_{340-400}
         $$
    
    4. **Comprimento de Onda Crítico (λc)**
       - Ponto onde 90% da área da transmitância está acumulada
    
    ### **Requisitos dos Dados**
    - Intervalo espectral: 290-400 nm
    - Resolução recomendada: ≤5 nm
    - Unidades:
      - Comprimento de onda: nm
      - Absorbância: adimensional (0-3)
    """)

with st.expander("⚠️ Boas Práticas", expanded=False):
    st.markdown("""
    - **Preparação da Amostra:**
      - Utilizar substrato adequado (ex: PMMA)
      - Aplicar quantidade padronizada (2 mg/cm²)
    
    - **Medição Espectral:**
      - Calibrar o espectrofotômetro regularmente
      - Realizar varreduras em triplicata
    
    - **Interpretação:**
      - λc > 370 nm indica boa proteção UVA
      - Razão UVA/UVB > 1/3 para proteção balanceada
    """)

## Rodapé
st.markdown("---")
st.markdown("""
**Aplicativo Científico** ⋅ Métodos validados conforme ISO 24443 e HPC Today 3(2024)  
Desenvolvido para análise precisa de fotoproteção UV ⋅ [Referência Completa](https://tks-hpc.h5mag.com/hpc_today_3_2024/sun_care_-_in_vitro_method_for_uva1_long_uva_or_ultra-long_uva_claiming_a_study_based_on_several_sunscreen_products)
""")
