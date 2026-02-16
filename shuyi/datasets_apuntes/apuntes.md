
## Algunos datasets: 

### Belebele(QA, zero-shot classification, text classification, multiple choice) -> Comprensión lectura(texto largo -> extrae información).

### COPA_es(text classification) -> Lógica(texto corto -> debe saber qué pasa luego según la lógica del mundo real, razonar la relación causal).
### fake_news_corpus_spanish(text classification) -> Clasificación booleana(clasifica una noticia si es verdadera/falsa).
### ClinDiagnosES(QA) -> Diagnosticar según el caso clínico.
### ClinTreatES(QA, text generation) -> Diagnosticar según el caso clínico y genera el tratamiento

### multilingual-crows-pairs -> Medir si el modelo tiene una porción alta de estereotipo(racista, sexista, clasista,...), detectarlo mediante el sesgo.

## 2 tipos de datasets:
### Commonsense Reasoning datasets: evalúa si el modelo tiene conocimientos básicos del mundo real, es como mide "anchura"(saber un poco de todo). Ejemplo: OpenBookQA.

### Multi-hop Reasoning datasets: evalúa si el modelo es capaz de conectar múltiples saltos para llegar a una conclusión compleja, es como mide "profundidad"(capacidad de razonamiento secuencial). Ejemplo: HotpotQA.


## Métricas:
### #Respuesta cerrada( NLU (Natural Language Understanding) ):
- EM(Exact Match): se usa en multiple-choice.
ejemplo:
Respuesta Correcta: "París"

Respuesta de la IA (Caso A): "París" → Puntuación: 1 (Éxito)

Respuesta de la IA (Caso B): "La ciudad de París" → Puntuación: 0 (Fallo)

Respuesta de la IA (Caso C): "Paris" (sin tilde) → Puntuación: 0 (Fallo)

- F1 Score: se obtiene alta puntuación si detecta palabras claves.
ejemplo:
Respuesta Correcta: "Hipertensión arterial sistémica"

Respuesta de la IA: "El paciente tiene hipertensión arterial"

Resultado: El EM sería 0 (no es idéntico), pero el F1 sería alto (aprox. 0.80) ya que detecta "hipertensión arterial" aunque le falta "sistemática".

### #Respuesta abierta ( NLG (Natural Language Generation) ):
- BLEU: busca coincidencias exactas entre la respuesta de IA y humana.(Ideal para traducción automática)
ejemplo:
Referencia humana: "El paciente debe tomar la medicación cada ocho horas."

Respuesta de la IA: "El paciente debe beber la medicina cada ocho horas."

Resultado: BLEU daría una puntuación baja/media. Aunque el mensaje es el mismo, las palabras "tomar" vs "beber" y "medicación" vs "medicina" no coinciden letra por letra.

- ROUGE: mide la proporcionalidad que consigue atrapar la IA.(Ideal para summarization y treatment)
ejemplo:
Referencia humana: "Es vital reposar, beber agua y tomar paracetamol."

Respuesta de la IA: "Debe tomar paracetamol y beber agua."

Resultado: ROUGE daría una puntuación alta. Aunque la IA olvidó el "reposar", capturó la mayoría de los elementos clave de la lista humana.

- METEOR: más flexible que BLEU ya que adapta a los sinónimos y variaciones.(Ideal para generación de texto)
ejemplo:
Referencia humana: "El niño corrió hacia la casa."

Respuesta de la IA: "El pequeño corría hacia el hogar."

Resultado: METEOR daría una puntuación muy alta. Entiende que "niño/pequeño", "corrió/corría" y "casa/hogar" significan lo mismo en este contexto.

-BERTScore: usa un modelo de tipo BERT para comparar el concepto de las frases.(Ideal para evaluar si la IA "entiende" lo que escribe)
ejemplo:
Frase 1: "La película fue un éxito total."

Frase 2: "El filme resultó ser un triunfo absoluto."

Resultado: BERTScore daría casi un 100%. Aunque no comparten casi ninguna palabra igual, la IA detecta que el vector de significado es idéntico.
