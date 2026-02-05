#!/bin/bash

if [ -t 0 ]; then
    if [ -f "$1" ]; then 
        INPUT=$(cat "$1") #Si es un fichero, leemos su contenido
    else 
        INPUT="$*" #Si no, usamos los argumentos como texto
    fi
else
    INPUT=$(cat -) #Si hay pipe, leemos la entrada estandar
fi


if [ -z "$INPUT" ]; then
    echo "USO: ./llama.sh [pregunta]"
    echo "O: echo [pregunta] | ./llama.sh"
    exit 1
fi

ollama run llama3.2 "$INPUT"

