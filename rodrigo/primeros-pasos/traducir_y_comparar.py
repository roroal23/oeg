import ollama

""" --- CONFIGURACION --- """
MODELO = 'llama3.2'
IDIOMA_ORIGEN = 'ingles'
SIGLAS_ORIGEN = 'eng'
IDIOMA_DESTINO = 'español'
SIGLAS_DESTINO = 'esp'
FICHERO = 'texto'
INST_TRADUCIR = f"""Eres un traductor profesional. Traduce el siguiente texto del {IDIOMA_ORIGEN} al {IDIOMA_DESTINO}. Mantén el formato
                    original y responde únicamente con la traducción."""

INST_COMPARAR = f"""A continuación te daré 2 textos: ambos fueron generados por {MODELO}. La principal diferencia es que 
                    el primero fue generado con la herramienta 'generate' y el segundo con la herramienta 'chat' (la orden 
                    descompuesta en un mensaje 'system' y uno 'user'). Menciona y explica el porque de las diferencias entre
                    ambos textos."""

""" --- FUNCIONES --- """
def read_file(filename: str) -> str:
    content = ''
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def write_file(filename: str, content: str):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


def translate_by_generate(text: str) -> str:
    response = ollama.generate(model = MODELO, prompt = f"{INST_TRADUCIR}\n{text}")
    return response['response']

def translate_by_chat(text: str) -> str:
    response = ollama.chat(model = MODELO, messages = [
        {'role': 'system', 'content': INST_TRADUCIR},
        {'role': 'user', 'content': text}
    ])
    return response['message']['content']

def explain_diff(text1: str, text2: str) -> str:
    response = ollama.chat(model = MODELO, messages = [
        {'role': 'system', 'content': INST_COMPARAR},
        {'role': 'user', 'content': text1},
        {'role': 'user', 'content': text2}
    ])
    return response['message']['content']

if __name__ == "__main__":
    FICHERO_ORIGEN = f'{FICHERO}-{SIGLAS_ORIGEN}.txt'
    FICHERO_DESTINO_GEN = f'{FICHERO}-{SIGLAS_DESTINO}-gen.txt'
    FICHERO_DESTINO_CHAT = f'{FICHERO}-{SIGLAS_DESTINO}-chat.txt'

    print("---> Leyendo fichero...")
    raw_text = read_file(FICHERO_ORIGEN)
    print("---> Iniciando traducciones...")

    translation_gen = translate_by_generate(raw_text)
    print("--> Traduccion usando generate:")
    print(translation_gen)

    translation_chat = translate_by_chat(raw_text)
    print("---> Traduccion usando chat:")
    print(translation_chat)

    print("---> Guardando en ficheros...")
    write_file(FICHERO_DESTINO_GEN, translation_gen)
    write_file(FICHERO_DESTINO_CHAT, translation_chat)

    print("---> Revisando diferencias en ficheros...")
    explanation = explain_diff(translation_gen, translation_chat)
    print(explanation)