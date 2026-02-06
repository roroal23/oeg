import ollama

def traducir_texto(entrada):
    traducido = ollama.chat(model = 'llama3.2', messages = [ {'role': 'system', 'content': 'Traducir el texto del usuario a español, solo quiero la frase traducida.'},
                                                             {'role': 'user', 'content': f'texto a traducir: "{entrada}"' }
                                                             ]
                            )
    return traducido['message']['content']

if __name__ == '__main__':
    mi_texto = "I am going to translate the sentence sent by the user."
    print(f"Entrada: {mi_texto}")
    print("-" * 10)
    print(f"Traducido: {traducir_texto(mi_texto)}")