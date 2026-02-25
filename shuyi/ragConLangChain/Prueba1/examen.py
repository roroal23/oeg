import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Cargar API KEY (En mi caso es de Groq)
load_dotenv()

# 1.Cargar el pdf
loader = PyPDFLoader("Discurso_Ingenieria_Ontologica.pdf")
paginas = loader.load_and_split()

# 2.Procesar el texto(Embeddings locales)
print("--- Analizando el PDF ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=paginas, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3.Configurar la IA (En mi caso Llama3 70B)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

# 4.Definir el formato del examen
template = """
Actúa como un profesor riguroso. Basándote en el siguiente contexto:
{context}

Crea un examen de 3 preguntas de opción múltiple (A, B, C) sobre los puntos clave.
Al final, muestra una sección de 'Respuestas Correctas'.
"""
prompt = ChatPromptTemplate.from_template(template)

# 5.Unir las piezas (Chain)
chain = (
	{"context": retriever, "question": RunnablePassthrough()}
	| prompt
	| llm
)

# 6.Ejecutar
print("--- Generando Examen ---")
resultado = chain.invoke("Genera un examen")
print(resultado.content)
