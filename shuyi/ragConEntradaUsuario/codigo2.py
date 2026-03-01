import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 1.Configuracion
st.set_page_config(page_title="Tutor RAG", layout="wide")
st.header("Tutor Inteligente: Sube tu PDF")
load_dotenv()

# Inicializar sesion
if "messages" not in st.session_state:
	st.session_state["messages"] = []
	st.session_state["examen_generado"] = None

# 2.Sidebar para subir el archivo
with st.sidebar:
	st.subheader("Documentacion")
	uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

# 3. Logica de procesamiento al subir un archivo
if uploaded_file and "retriever" not in st.session_state:
		# Guardar en archivo temporal
		with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
			tmp.write(uploaded_file.getvalue())
			tmp_path = tmp.name

		# Procesar PDF
		with st.spinner("Analizando documento..."):
			loader = PyPDFLoader(tmp_path)
			paginas = loader.load_and_split()
			embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
			vectorstore = Chroma.from_documents(documents=paginas, embedding=embeddings)
			st.session_state["retriever"] = vectorstore.as_retriever()
			os.remove(tmp_path) # limpiar el archivo temporal
		st.success("Documento procesado correctamente.")

# 4.Generacion del examen
if "retriever" in  st.session_state and st.session_state["examen_generado"] is None:
	if st.button("Generar Examen"):
		with st.spinner("Creando examen..."):
			llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

			context = st.session_state["retriever"].invoke("puntos clave del documento")
		prompt = f"""
			Basado en este contexto: {context}, genera 3 preguntas de opcion multiple (A, B, C).
			Devuelve el resultado exactamente en este formato:

			### PREGUNTAS
			1. [Pregunta] A) B) C)
			2. [Pregunta] A) B) C)
			3. [Pregunta] A) B) C)

			### OPCION_CORRECTA
			1. [Letra]
			2. [Letra]
			3. [Letra]
			"""
		response = llm.invoke(prompt)

		st.session_state["examen_generado"] = response.content
		st.rerun()

# 5.Mostrar el examen y chat
if st.session_state["examen_generado"]:
	contenido = st.session_state["examen_generado"]

	if "OPCION_CORRECTA" in contenido:
		partes = contenido.split("### OPCION_CORRECTA")
		preguntas_solo = partes[0]
		clave_solo = partes[1]
	else:
		# Si falla ocultar la solución, dejamos que muestre todo.
		preguntas_solo = contenido
		clave_solo = "No se encontró clave"
		st.warning("El modelo no generó el formato correctamente")

	st.markdown(preguntas_solo)

	with st.form("examen_form"):
		respuestas_usuario = st.text_area("Escribe tus respuestas (Ej: 1:A, 2:B, 3:C)")
		submit = st.form_submit_button("Enviar")

	if submit:
		st.write("### Evaluando...")

		eval_prompt = f"""
		La clave de respuestas es: {clave_solo}
		El usuario ha respondido: {respuestas_usuario}
		Compara las respuestas del usuario con la clave de respuestas y dime si acertó cada una.
		"""

		llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

		response = llm.invoke(eval_prompt)

		st.success("Resultados:")
		st.write(response.content)
