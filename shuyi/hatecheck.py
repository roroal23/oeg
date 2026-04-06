import os
import pandas as pd
from transformers import pipeline
from datasets import load_dataset

# Configuracion de rutas
os.environ['TRANSFORMERS_CACHE'] = '/home/x244/SCRATCH/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/home/x244/SCRATCH/hf_cache'

def run_audit():
    # 1. Cargar el dataset específico de HateCheck en español
    print("Cargando HateCheck-ES...")
    dataset = load_dataset("Paul/hatecheck-spanish", split="test")

    # 2. Filtrar solo por discapacidad
    df = pd.DataFrame(dataset)
    df_discap = df[df['target_ident'] == 'disabled people'].copy()
    print(f"Total de casos detectados para discapacidad: {len(df_discap)}")

    # 3. Cargar el modelo BETO en la GPU
    print("Cargando BETO en la GPU...")
    pipe = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis", device=0)

    # 4. Procesar las frases
    print("Analizando sesgos... Esto puede tardar un minuto.")
    
    def predict_beto(text):
        res = pipe(text)[0]
        return res['label'], res['score']

    # Aplicamos la predicción
    results = df_discap['test_case'].apply(lambda x: pd.Series(predict_beto(x)))
    df_discap[['pred_label', 'pred_score']] = results

    # 5. Identificar errores éticos (Sesgos)
    # Un sesgo ocurre si la frase es 'non-hateful' (humano) pero la IA dice 'NEG' (o 'NEU' con baja confianza)
    # En este modelo de sentimiento, comparamos 'non-hateful' vs 'POS/NEU'
    df_discap['is_bias'] = (df_discap['label_gold'] == 'non-hateful') & (df_discap['pred_label'] == 'NEG') | (df_discap['label_gold'] == 'hateful') & ( (df_discap['pred_label'] == 'POS') | (df_discap['pred_label'] == 'NEU') )

    # 6. Guardar resultados detallados en un CSV
    output_path = '/home/x244/SCRATCH/test_etica/resultados_audit_hatecheck.csv'
    df_discap.to_csv(output_path, index=False)
    print(f"\nResultados detallados guardados en: {output_path}")

    # 7. Resumen de la Auditoría
    total_casos = len(df_discap)
    total_sesgos = df_discap['is_bias'].sum()
    porcentaje_sesgo = (total_sesgos / total_casos) * 100

    print("\n" + "="*40)
    print("RESUMEN DE AUDITORÍA ÉTICA (DISCAPACIDAD)")
    print("="*40)
    print(f"Casos analizados: {total_casos}")
    print(f"Sesgos detectados: {total_sesgos}")
    print(f"Tasa de error por sesgo: {porcentaje_sesgo:.2f}%")
    print("="*40)

    # Mostrar algunos ejemplos de sesgo si existen
    if total_sesgos > 0:
        print("\nEjemplos donde la IA falló:")
        ejemplosNEG = df_discap[(df_discap['is_bias'] == True) & (df_discap['pred_label'] == 'NEG')].head(5)
        for _, row in ejemplosNEG.iterrows():
            print(f"- Frase: {row['test_case']}")
            print(f"  IA dijo: {row['pred_label']} ({row['pred_score']:.3f})\n")
        ejemplosPOS_NEU = df_discap[(df_discap['is_bias'] == True) & ((df_discap['pred_label'] == 'POS') | (df_discap['pred_label'] == 'NEU'))].head(5)
        for _, row in ejemplosPOS_NEU.iterrows():
            print(f"- Frase: {row['test_case']}")
            print(f"  IA dijo: {row['pred_label']} ({row['pred_score']:.3f})\n")

if __name__ == "__main__":
    run_audit()
