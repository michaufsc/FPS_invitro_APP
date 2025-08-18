import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from io import BytesIO

# Configuração da página
st.set_page_config(
    page_title="Análise de Fotoproteção",
    page_icon="🌞",
    layout="wide"
)

# Título e descrição
st.title("🌞 Análise de Fotoproteção In Vitro")
st.markdown("""
Aplicativo para cálculo de parâmetros de proteção solar a partir de dados espectrais.
""")

# =============================================
# FUNÇÕES PRINCIPAIS
# =============================================

def criar_template():
    """Cria um template Excel com dados de exemplo"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Dados UVB
        uvb_data = pd.DataFrame({
            'Comprimento de Onda (nm)': range(290, 321, 5),
            'Absorbância': np.linspace(0.1, 1.0, 7),
            'E(λ)': [0.0150, 0.0817, 0.2874, 0.3278, 0.1864, 0.0839, 0.0180],
            'I(λ)': [0.134, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139]
        })
        uvb_data.to_excel(writer, sheet_name='UVB', index=False)
        
        # Dados UVA
        uva_data = pd.DataFrame({
            'Comprimento de Onda (nm)': range(320, 401, 5),
            'Absorbância': np.linspace(0.5, 2.0, 17),
            'P(λ)': np.linspace(0.1, 0.8, 17),
            'I(λ)': np.linspace(0.1, 0.9, 17)
        })
        uva_data.to_excel(writer, sheet_name='UVA', index=False)
        
        # Instruções
        instrucoes = pd.DataFrame({
            'INSTRUÇÕES': [
                '1. Este arquivo contém duas abas obrigatórias: UVB e UVA',
                '2. Na aba UVB (290-320nm) mantenha as colunas exatas:',
                '   - Comprimento de Onda (nm), Absorbância, E(λ), I(λ)',
                '3. Na aba UVA (320-400nm) mantenha as colunas exatas:',
                '   - Comprimento de Onda (nm), Absorbância, P(λ), I(λ)',
                '4. Substitua APENAS a coluna "Absorbância" por seus dados experimentais',
                '5. Não altere os nomes das abas ou cabeçalhos'
            ]
        })
        instrucoes.to_excel(writer, sheet_name='LEIA-ME', index=False)
    
    return output.getvalue()

def calcular_constante_c(df_uvb, spf_in_vivo):
    """Calcula a constante C que relaciona dados in vitro/in vivo"""
    def erro(C):
        return abs(calcular_spf(df_uvb, C) - spf_in_vivo)
    resultado = opt.minimize_scalar(erro, bounds=(0.5, 1.5), method='bounded')
    return resultado.x

def calcular_spf(df, C=1.0, lambda_min=290, lambda_max=320):
    """Calcula SPF para faixa espectral definida"""
    df_faixa = df[(df['Comprimento de Onda (nm)'] >= lambda_min) & 
                (df['Comprimento de Onda (nm)'] <= lambda_max)].copy()
    num = np.trapz(df_faixa['E(λ)'] * df_faixa['I(λ)'], 
                  df_faixa['Comprimento de Onda (nm)'])
    den = np.trapz(df_faixa['E(λ)'] * df_faixa['I(λ)'] * 10**(-df_faixa['Absorbância'] * C), 
                  df_faixa['Comprimento de Onda (nm)'])
    return num / den

def calcular_uva_pf(df_uva, C):
    """Calcula UVA-PF conforme método ISO"""
    df_faixa = df_uva[(df_uva['Comprimento de Onda (nm)'] >= 320) & 
                     (df_uva['Comprimento de Onda (nm)'] <= 400)].copy()
    num = np.trapz(df_faixa['P(λ)'] * df_faixa['I(λ)'], 
                  df_faixa['Comprimento de Onda (nm)'])
    den = np.trapz(df_faixa['P(λ)'] * df_faixa['I(λ)'] * 10**(-df_faixa['Absorbância'] * C), 
                  df_faixa['Comprimento de Onda (nm)'])
    return num / den

def calcular_cwc(df):
    """Calcula Comprimento de Onda Crítico"""
    df_faixa = df[(df['Comprimento de Onda (nm)'] >= 290) & 
                (df['Comprimento de Onda (nm)'] <= 400)].copy()
    area_total = np.trapz(df_faixa['Absorbância'], 
                         df_faixa['Comprimento de Onda (nm)'])
    area_cum = np.cumsum(df_faixa['Absorbância'] * np.gradient(df_faixa['Comprimento de Onda (nm)']))
    idx = np.where(area_cum >= 0.9 * area_total)[0][0]
    return df_faixa['Comprimento de Onda (nm)'].iloc[idx]

def calcular_mansur(df_uvb):
    """Calcula SPF pelo método de Mansur"""
    df = df_uvb.copy()
    df['EE_I'] = df['E(λ)'] * df['I(λ)']
    df['Contribuição'] = df['EE_I'] * df['Absorbância']
    return 10 * df['Contribuição'].sum()

# =============================================
# INTERFACE DO USUÁRIO
# =============================================

with st.sidebar:
    st.header("📤 Upload de Dados")
    uploaded_file = st.file_uploader("Carregue seu arquivo Excel", type=["xlsx"])
    
    if st.button("⬇️ Baixar Template"):
        template = criar_template()
        st.download_button(
            label="Template_Fotoproteção.xlsx",
            data=template,
            file_name="Template_Fotoproteção.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.header("⚙️ Configurações")
    metodo = st.selectbox(
        "Método de Cálculo",
        ["ISO 24443 (SPF)", "Método de Mansur", "CWC", "UVA-PF", "Análise Completa"]
    )
    
    if metodo in ["ISO 24443 (SPF)", "UVA-PF", "Análise Completa"]:
        spf_in_vivo = st.number_input("SPF in vivo conhecido", min_value=1.0, value=30.0, step=0.5)

# =============================================
# PROCESSAMENTO PRINCIPAL
# =============================================

if uploaded_file is not None:
    try:
        # Carregar dados
        df_uvb = pd.read_excel(uploaded_file, sheet_name='UVB')
        df_uva = pd.read_excel(uploaded_file, sheet_name='UVA')
        
        # Verificar colunas
        cols_uvb = ['Comprimento de Onda (nm)', 'Absorbância', 'E(λ)', 'I(λ)']
        cols_uva = ['Comprimento de Onda (nm)', 'Absorbância', 'P(λ)', 'I(λ)']
        
        if not all(col in df_uvb.columns for col in cols_uvb):
            st.error("❌ Colunas faltando na aba UVB")
        elif not all(col in df_uva.columns for col in cols_uva):
            st.error("❌ Colunas faltando na aba UVA")
        else:
            st.success("✅ Arquivo validado com sucesso!")
            
            # Calcular constante C se necessário
            C = calcular_constante_c(df_uvb, spf_in_vivo) if 'spf_in_vivo' in locals() else 1.0
            
            # Resultados
            resultados = {}
            
            if metodo in ["ISO 24443 (SPF)", "Análise Completa"]:
                spf = calcular_spf(df_uvb, C)
                resultados["SPF (ISO 24443)"] = spf
                resultados["Constante C"] = C
            
            if metodo in ["Método de Mansur", "Análise Completa"]:
                spf_mansur = calcular_mansur(df_uvb)
                resultados["SPF (Mansur)"] = spf_mansur
            
            if metodo in ["CWC", "Análise Completa"]:
                df_completo = pd.concat([df_uvb, df_uva])
                cwc = calcular_cwc(df_completo)
                resultados["CWC (nm)"] = cwc
                resultados["Proteção UVA"] = "✅ Adequada" if cwc >= 370 else "⚠️ Insuficiente"
            
            if metodo in ["UVA-PF", "Análise Completa"]:
                uva_pf = calcular_uva_pf(df_uva, C)
                resultados["UVA-PF"] = uva_pf
                if 'spf' in resultados:
                    resultados["Razão UVA/SPF"] = uva_pf / resultados["SPF (ISO 24443)"]
            
            # Exibir resultados
            st.subheader("📊 Resultados")
            for param, valor in resultados.items():
                if isinstance(valor, float):
                    st.metric(param, f"{valor:.2f}")
                else:
                    st.metric(param, valor)
            
            # Gráfico
            st.subheader("📈 Espectro de Absorbância")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_uvb['Comprimento de Onda (nm)'], df_uvb['Absorbância'], label='UVB')
            ax.plot(df_uva['Comprimento de Onda (nm)'], df_uva['Absorbância'], label='UVA')
            
            if metodo == "CWC" or (metodo == "Análise Completa" and 'CWC (nm)' in resultados):
                ax.axvline(x=cwc, color='r', linestyle='--', label=f'CWC = {cwc:.1f} nm')
            
            ax.set_xlabel("Comprimento de Onda (nm)")
            ax.set_ylabel("Absorbância")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Exportar resultados
            st.subheader("📤 Exportar Resultados")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pd.DataFrame.from_dict(resultados, orient='index', columns=['Valor']).to_excel(writer, sheet_name='Resultados')
                df_uvb.to_excel(writer, sheet_name='Dados_UVB', index=False)
                df_uva.to_excel(writer, sheet_name='Dados_UVA', index=False)
            
            st.download_button(
                label="⬇️ Baixar Relatório Completo",
                data=output.getvalue(),
                file_name="Resultados_Fotoproteção.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {str(e)}")
        st.error("Verifique se o arquivo segue exatamente o formato do template")

else:
    st.info("""
    **Instruções:**
    1. Baixe o template clicando no botão ao lado
    2. Substitua APENAS a coluna 'Absorbância' por seus dados experimentais
    3. Faça upload do arquivo modificado
    4. Selecione o método de análise desejado
    """)
    st.image("https://via.placeholder.com/600x200?text=Exemplo+de+Layout+do+Arquivo", width=600)
