# Ideas básicas de técnicas de evaluación

## LLM-as-judge
Usa un LLM de nivel superior (al menos con  la misma capacidad que el modelo a ser evaluado) para evaluar las respuestas
-  Temperatura = 0 -> siempre elige la respuesta más probable, con lo que, de cierta manera, el resultado es determinista
- Pruebas de permutación (Shuffle):
  (pregunta, opción A, opción B) + (pregunta, opción B, opción A)
    - Los modelos suelen preferir la 1ra respuesta 
    - Estrategia: Si ambas permutaciones dan la misma respuesta se considera valida, sino se considera inconsistente
- Few-shot examples: Proporcionar ejemplos sencillos de la entrada y salida, esperada, luego preguntar sobre un caso concreto. 1-shot example = 1 ejemplo. Formato consistente. Ejemplo representativos
- Fiabilidad inter modelo
    1. Tomar una muestra (50 ejemplos)
    2. Pasar la muestra por el LLM varias veces en distintos momentos
    3. Calcular correlación o acuerdo de Kappa
- Comparación con profesionales: linguista

## Task-based
(Lo que se ve en HuggingFace La Leaderboard)
- Traducción inversa: Traducir textos para ver si...
    1. Traduce palabra por palabra (traducción plana)
    2. Pierde el contexto de la frase
    3. Mantiene correción sintáctica
- Alucinaciones linguísticas:
    1. Usar "false friends"
    2. Forzar al modelo para que genere palabras inexistentes

## Evaluación métrica
- BLEU/ROUGE/METEOR : Comparar la respuesta del LLM con una respuesta humana, palabra por palabra
- BERTScore : Usa embeddings para medir la similitud semántica


# Búsqueda [En construcción]

# Métricas de evaluación

## BLEU (Bilingual Evaluation Understudy)
BLEU es una métrica que se creó, originalmente, para comparar una traducción candidata (hecha por un modelo) frente a una o varias traducciones de referencia (hecha por humanos). 
- Aunque fue creada para evaluar traducciones, se puede usar para cualquier tarea que requiere comparar una respuesta generada y una humana. Ej: resumir, parafrasear
- Es una forma rápida y muy poco costosa de evaluar

Esta métrica, a groso modo, evalúa cuantas palabras del texto candidato aparecen dentro de los textos de referencia. Esta toma en cuenta:
- Número de N-gramas del candidato que aparecen en los candidatos. Haciendo una proporción de estos respecto a la longitud del candidato. 
- Diferencia entre la longitud del candidato y las referencias. Penalizando por traducciones con longitudes menores a las de referencia. **Brevity Penalty**

### N-gramas
Un N-grama es la conjunción de N palabras. Por ejemplo, en "El perro come pienso" tendríamos 1-gramas={El, perro, come, pienso}, 2-gramas={El perro, perro come, come pienso}, 3-gramas={El perro come, perro come pienso}

En esta métrica, N es un parámetro. Una N pequeña se enfoca en evaluar el significado de las oraciones, mientras que una N grande se enfoca en las oraciones bien formadas.

Implementación:

nltk.translate.bleu_score

Fuentes:
- https://aclanthology.org/P02-1040.pdf
- https://medium.com/nlplanet/two-minutes-nlp-learn-the-bleu-metric-by-examples-df015ca73a86
- https://www.ibm.com/docs/en/watsonx/saas?topic=metrics-bleu
- https://www.geeksforgeeks.org/nlp/nlp-bleu-score-for-evaluating-neural-machine-translation-python/
- https://www.nltk.org/api/nltk.translate.bleu_score.html

## ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
De forma similar a BLEU, REOUGE es una serie de métricas que evaluan la similitud entre un texto generado y otro de referencia.
- Se enfoca más en la sintaxis que en la semántica.

### ROUGE-N
La métrica ROUGE-N evalúa el solapamiento de N-gramas entre el texto generado y el de referencia. Es decir, el número de palabras consecutivas  que aparecen tanto en el texto generado como en el texto de referencia.
Se calcula como:
ROUGE[n]F1 = ( 2* ROUGE[n]recall) / (ROUGE[n]recall + ROUGE[n]precision)

### ROUGE-L
La métrica ROUGE-L considera la "longest common subsequence of words (LCS)", es decir, la secuencia (no necesariamente consecutiva) de palabras más larga.
Se calcula obteniendo: ROUGE-L-F1

### ROUGE-S
Esta métrica considera a los skip-gram, que son N-gramas que aparecen tanto en el texto generado como en el de referencia y a los que se les permite estar separados por una o más palabras en el texto generado.
Por ejemplo: "Mi perro come carne" y "Mi perro come pienso de carne" coincidirán si se usa un 2-gram skipping

Implementación:

rouge_score

Fuentes:
- https://www.ibm.com/docs/en/waasfgm?topic=metrics-rouge
- https://www.traceloop.com/blog/evaluating-model-performance-with-the-rouge-metric-a-comprehensive-guide
- https://pypi.org/project/rouge-score/

## METEOR (Metric for Evaluation of Translation with Explicit ORdering)
Esta métrica representa una mejora respecto a otras como BLEU porque incorpora a su analisis características del lenguaje como sinonimia, stemming (palabras que se generan a partir de una misma raíz), orden de palabras (similar a ROUGE).
Además, da mayor peso al recall para alinearse mejor al juicio humano sobre la calidad de una traducción

- METEOR da mayor peso al recall, lo que hace más sensible a traducciones que capturan el significado en conjunto de la referencia, aunque no hayan traducido palabra por palabra.
- METEOR incorpora mecanismos que analizan sinonimos y stemmings (palabras que vienen de la misma raíz), lo que permite apreciar palabrar cercanas semanticamente. Por ejemplo, gato y gatos, o gato y felino. 
- Penaliza el alineamiento no contiguo de palabras, es decir, que un grupo de palabras esten en posiciones distintas al texto de referencia. Ej: "Ellos estuvieron jugando en la montaña" y "En la montaña, ellos estuvieron jugando". Esto favorece a traducciones que conservan la estructura del texto de referencia, manteniendo el orden de palabras y la coherencia.

- Es mucho más costoso que métodos como BLEU
- La penalización por la alineación no contigua de palabras puede ser contraproducente, porque pueden existir varias formas correctas de ordenar una idea.

Implementación:

nltk.translate.meteor

Fuentes:

- https://medium.com/on-being-ai/what-is-meteor-metric-for-evaluation-of-translation-with-explicit-ordering-45b49ac5ec70