import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Cargar API KEY(en archivo .env)
load_dotenv("./resources/.env")

st.set_page_config(
    page_title = "RAG",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.header("RAG")

st.markdown("En este chat se le brinda al modelo un fichero")


if st.button("Generar Examen"):
	with st.spinner("Procesando documento y generando examen..."):
		loader = PyPDFLoader("./resources/Discurso_Ingenieria_Ontologica.pdf")
		paginas = loader.load_and_split()

		embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
		vectorstore = Chroma.from_documents(documents=paginas, embedding=embeddings)
		retriever = vectorstore.as_retriever()

		llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

		template = """
		Actúa como un profesor riguroso. Basándote en el siguiente contexto:
		{context}

		Crea un examen de 3 preguntas de opción múltiple (A, B, C) sobre los puntos clave.
		Al final, muestra una sección de 'Respuestas Correctas'.
		"""
		prompt = ChatPromptTemplate.from_template(template)

		chain = (
        		{"context": retriever, "question": RunnablePassthrough()}
        		| prompt
        		| llm
		)

		resultado = chain.invoke("Genera un examen")

		st.markdown("---")
		st.markdown(resultado.content)
