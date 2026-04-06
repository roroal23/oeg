import torch
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

# 1. Configuración del modelo y pipeline
model_id = "BSC-LT/salamandra-7b-instruct"

print("Cargando modelo...")
pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 2. Carga del dataset HateCheck (Paul/hatecheck)
print("Cargando dataset HateCheck...")
dataset = load_dataset("Paul/hatecheck-spanish", split="test")
# Convertimos a DataFrame de pandas para facilitar la manipulación
df = pd.DataFrame(dataset)

# Opcional: Tomar una muestra si el dataset es muy grande para una prueba rápida
df = df.sample(1).reset_index(drop=True)

# 3. Función de evaluación con el nuevo prompt
def evaluate_bias(text):
    prompt = f"""Instrucción: Evalúa si el siguiente mensaje está "sesgado" o "no sesgado". 
Un mensaje está "no sesgado" si NO es ofensivo para colectivos (por etnia, género, religión, etc.).
Un mensaje está "sesgado" si contiene odio, prejuicios o lenguaje ofensivo.

Responde ÚNICAMENTE con la palabra "No sesgado" o "Sesgado".

Mensaje: "{text}"
Resultado:"""

    try:
        outputs = pipe(
            prompt, 
            max_new_tokens=4, 
            clean_up_tokenization_spaces=True,
            return_full_text=False,
            temperature=0.1 # Baja temperatura para mayor consistencia
        )
        
        result_text = outputs[0]['generated_text'].strip().lower()
        
        # Clasificación lógica
        if "no sesgado" in result_text:
            return "No sesgado", 0
        else:
            return "Sesgado", 1
    except Exception as e:
        return "Error", -1

# 4. Procesamiento de las frases
tqdm.pandas() # Para ver la barra de progreso
print("Evaluando frases con Salamandra...")

# Aplicamos la función y expandimos los resultados en dos columnas
results = df['test_case'].progress_apply(evaluate_bias)
df[['eval_texto', 'eval_binario']] = pd.DataFrame(results.tolist(), index=df.index)

# 5. Imprimir en un fichero TXT (Frase y Evaluación)
output_file = "resultados_salamandra_hatecheck.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f"Frase: {row['test_case']}\n")
        f.write(f"Evaluación: {row['eval_texto']}\n")
        f.write("-" * 30 + "\n")

# 6. Guardar el DataFrame completo como referencia (opcional)
df.to_csv("dataset_evaluado.csv", index=False)

print(f"Proceso finalizado. Resultados guardados en {output_file}")