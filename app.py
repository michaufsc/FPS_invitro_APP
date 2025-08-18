import pandas as pd
import numpy as np
from scipy.integrate import simps
from scipy.optimize import bisect
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Dados de referência da norma ISO 24443 (Anexo C)
PPD_ACTION_SPECTRUM = {
    320: 1.000, 330: 0.750, 340: 0.500, 350: 0.438, 360: 0.376,
    370: 0.314, 380: 0.252, 390: 0.190, 400: 0.128
}

ERYTHEMA_ACTION_SPECTRUM = {
    290: 1.000, 300: 0.649, 310: 0.074, 320: 0.0086, 330: 0.0014,
    340: 0.00097, 350: 0.00068, 360: 0.00048, 370: 0.00031, 380: 0.00025, 400: 0.00012
}

UV_SSR_IRRADIANCE = {
    290: 8.741E-06, 300: 1.478E-02, 320: 0.7236, 340: 0.9939, 
    360: 1.037, 380: 0.5341, 400: 0.0042
}

UVA_IRRADIANCE = {
    320: 4.843E-06, 340: 5.198E-04, 360: 1.078E-03, 
    380: 7.105E-04, 400: 1.045E-05
}

class UVAAnalyzer:
    def __init__(self):
        self.results = {}
        self.plate_data = {}
        
    def load_data(self, file_path):
        """Carrega dados de arquivo CSV ou Excel"""
        try:
            if file_path.endswith('.csv'):
                return self._process_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return self._process_excel(file_path)
            else:
                raise ValueError("Formato de arquivo não suportado")
        except Exception as e:
            raise ValueError(f"Erro ao processar arquivo: {str(e)}")
    
    def _process_csv(self, file_path):
        """Processa arquivos no formato CSV"""
        df = pd.read_csv(file_path, sep=';', decimal=',')
        
        # Verificar colunas necessárias
        required_cols = ['wavelength', 'P', 'I', 'A_e']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Estrutura do CSV inválida. Colunas esperadas: wavelength;P;I;A_e")
        
        # Converter dados numéricos (tratando vírgula decimal)
        data = {
            'wavelength': pd.to_numeric(df['wavelength'].astype(str).str.replace(',', '.')),
            'PPD_action': pd.to_numeric(df['P'].astype(str).str.replace(',', '.')),
            'irradiance': pd.to_numeric(df['I'].astype(str).str.replace(',', '.')),
            'absorbance_post': pd.to_numeric(df['A_e'].astype(str).str.replace(',', '.'))
        }
        
        # Verificar faixa espectral
        self._validate_spectral_range(data['wavelength'])
        
        return data
    
    def _process_excel(self, file_path):
        """Processa arquivos no formato Excel"""
        df = pd.read_excel(file_path)
        
        # Normalizar nomes de colunas
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Mapear colunas possíveis
        col_mapping = {
            'comprimento de onda': 'wavelength',
            'e(λ)': 'erythema_action',
            'i(λ)': 'irradiance',
            'absorbancia': 'absorbance'
        }
        
        # Renomear colunas
        df = df.rename(columns={k:v for k,v in col_mapping.items() if k in df.columns})
        
        # Verificar colunas necessárias
        required_cols = ['wavelength', 'absorbance']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Estrutura do Excel inválida")
        
        data = {
            'wavelength': pd.to_numeric(df['wavelength'].astype(str).str.replace(',', '.')),
            'absorbance': pd.to_numeric(df['absorbance'].astype(str).str.replace(',', '.'))
        }
        
        # Adicionar dados opcionais se existirem
        if 'erythema_action' in df.columns:
            data['erythema_action'] = pd.to_numeric(df['erythema_action'].astype(str).str.replace(',', '.'))
        if 'irradiance' in df.columns:
            data['irradiance'] = pd.to_numeric(df['irradiance'].astype(str).str.replace(',', '.'))
        
        # Verificar faixa espectral
        self._validate_spectral_range(data['wavelength'])
        
        return data
    
    def _validate_spectral_range(self, wavelengths):
        """Valida se os dados cobrem a faixa espectral necessária"""
        if min(wavelengths) > 290 or max(wavelengths) < 400:
            raise ValueError("Dados devem cobrir a faixa de 290-400nm")
    
    def calculate_protection_factors(self, data, in_vivo_spf, plate_id=None):
        """Calcula todos os parâmetros de proteção UVA"""
        try:
            # Preparar dados interpolados para cálculos
            wavelengths = np.arange(290, 401)  # 290-400nm em passos de 1nm
            
            # Interpolar absorbância para todos os comprimentos de onda
            absorbance = np.interp(wavelengths, data['wavelength'], data['absorbance'])
            
            # 1. Calcular SPF in vitro inicial
            erythema = np.interp(wavelengths, 
                               list(ERYTHEMA_ACTION_SPECTRUM.keys()),
                               list(ERYTHEMA_ACTION_SPECTRUM.values()))
            
            irradiance = np.interp(wavelengths,
                                 list(UV_SSR_IRRADIANCE.keys()),
                                 list(UV_SSR_IRRADIANCE.values()))
            
            transmittance = 10 ** (-absorbance)
            
            numerator = simps(erythema * irradiance, wavelengths)
            denominator = simps(erythema * irradiance * transmittance, wavelengths)
            spf_in_vitro = numerator / denominator
            
            # 2. Calcular fator de ajuste C
            def objective(c):
                adjusted_trans = 10 ** (-absorbance * c)
                adj_denominator = simps(erythema * irradiance * adjusted_trans, wavelengths)
                return (numerator / adj_denominator) - in_vivo_spf
            
            try:
                c = bisect(objective, 0.8, 1.6)
            except ValueError:
                raise ValueError("Não foi possível encontrar fator C válido (0.8-1.6). Verifique os dados de entrada.")
            
            # 3. Calcular UVA-PF
            ppd = np.interp(wavelengths,
                           list(PPD_ACTION_SPECTRUM.keys()),
                           list(PPD_ACTION_SPECTRUM.values()))
            
            uva_irradiance = np.interp(wavelengths,
                                     list(UVA_IRRADIANCE.keys()),
                                     list(UVA_IRRADIANCE.values()))
            
            uva_numerator = simps(ppd * uva_irradiance, wavelengths)
            uva_denominator = simps(ppd * uva_irradiance * (10 ** (-absorbance * c)), wavelengths)
            uva_pf = uva_numerator / uva_denominator
            
            # 4. Calcular dose de exposição
            exposure_dose = 1.2 * uva_pf
            
            # Armazenar resultados
            result = {
                'SPF_in_vitro': round(spf_in_vitro, 2),
                'Adjustment_factor_C': round(c, 3),
                'UVA_PF': round(uva_pf, 2),
                'Exposure_dose_Jcm2': round(exposure_dose, 2),
                'wavelengths': wavelengths,
                'absorbance': absorbance,
                'adjusted_absorbance': absorbance * c
            }
            
            if plate_id:
                self.plate_data[plate_id] = result
            
            return result
            
        except Exception as e:
            raise ValueError(f"Erro nos cálculos: {str(e)}")
    
    def generate_report(self, results, output_dir='reports'):
        """Gera relatório PDF com gráficos e resultados"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"uva_report_{timestamp}.pdf")
            
            # Criar gráficos
            plt.figure(figsize=(12, 8))
            
            # Gráfico 1: Absorbância vs Comprimento de Onda
            plt.subplot(2, 2, 1)
            plt.plot(results['wavelengths'], results['absorbance'], 'b-', label='Absorbância medida')
            plt.plot(results['wavelengths'], results['adjusted_absorbance'], 'r--', label='Absorbância ajustada (×C)')
            plt.xlabel('Comprimento de Onda (nm)')
            plt.ylabel('Absorbância')
            plt.title('Perfil de Absorbância')
            plt.legend()
            plt.grid(True)
            
            # Gráfico 2: Espectros de Ação
            plt.subplot(2, 2, 2)
            plt.plot(results['wavelengths'], 
                    np.interp(results['wavelengths'], 
                             list(ERYTHEMA_ACTION_SPECTRUM.keys()),
                             list(ERYTHEMA_ACTION_SPECTRUM.values())),
                    'r-', label='Eritema (E(λ))')
            plt.plot(results['wavelengths'], 
                    np.interp(results['wavelengths'], 
                             list(PPD_ACTION_SPECTRUM.keys()),
                             list(PPD_ACTION_SPECTRUM.values())),
                    'b-', label='PPD (P(λ))')
            plt.xlabel('Comprimento de Onda (nm)')
            plt.ylabel('Sensibilidade')
            plt.title('Espectros de Ação')
            plt.legend()
            plt.grid(True)
            
            # Gráfico 3: Transmitância
            plt.subplot(2, 2, 3)
            transmittance = 10 ** (-results['adjusted_absorbance'])
            plt.plot(results['wavelengths'], transmittance * 100, 'g-')
            plt.xlabel('Comprimento de Onda (nm)')
            plt.ylabel('Transmitância (%)')
            plt.title('Perfil de Transmitância')
            plt.grid(True)
            
            # Texto com resultados
            plt.subplot(2, 2, 4)
            plt.axis('off')
            report_text = (
                f"RESULTADOS DO TESTE UVA (ISO 24443)\n\n"
                f"SPF in vitro calculado: {results['SPF_in_vitro']}\n"
                f"Fator de ajuste C: {results['Adjustment_factor_C']}\n"
                f"UVA-PF: {results['UVA_PF']}\n"
                f"Dose de exposição: {results['Exposure_dose_Jcm2']} J/cm²\n\n"
                f"Data do teste: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
            )
            plt.text(0.1, 0.5, report_text, fontsize=12)
            
            plt.tight_layout()
            plt.savefig(report_path)
            plt.close()
            
            return report_path
            
        except Exception as e:
            raise ValueError(f"Erro ao gerar relatório: {str(e)}")

# Exemplo de uso:
if __name__ == "__main__":
    analyzer = UVAAnalyzer()
    
    try:
        # Carregar dados (usar o caminho real do arquivo)
        data = analyzer.load_data('1_A2_UVA_test.CSV')  # ou '1_A_1.xlsx'
        
        # Calcular parâmetros (fornecer o SPF in vivo conhecido)
        results = analyzer.calculate_protection_factors(data, in_vivo_spf=30)
        
        # Gerar relatório
        report_path = analyzer.generate_report(results)
        
        print(f"Análise concluída com sucesso! Relatório gerado em: {report_path}")
        print("\nResultados:")
        for key, value in results.items():
            if not isinstance(value, np.ndarray):  # Não mostrar arrays grandes
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Erro: {str(e)}")
