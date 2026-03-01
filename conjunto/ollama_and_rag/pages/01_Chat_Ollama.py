import streamlit as st
import ollama

st.set_page_config(
    page_title = "Chat Ollama",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.header("Chat Ollama")

st.markdown("""En este chat se usa el modelo Qwen 2.5 de 7 billones de parametros junto con la prompt:
            
            Tu función es explicar conceptos complejos en tres niveles distintos:
            1. Como si fueras un niño (con vocabulario muy básico)
            2. Como si fueras un académico de Harvard
            3. Usando únicamente emojis.
            Presenta tus explicaciones de forma clara y concisa.""")



if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append(
        {"role":    "system",
         "content": """Tu función es explicar conceptos complejos en tres niveles distintos:
            1. Como si fueras un niño (con vocabulario muy básico), 2. Como si fueras un académico de Harvard, 3. Usando únicamente emojis.
            Presenta tus explicaciones de forma clara y concisa. """}
    )

with st.container():
    for message in st.session_state["messages"]:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

prompt = st.chat_input("Introduzca el concepto a explicar")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        stream = response = ollama.chat(model = "qwen2.5:7b", messages = st.session_state["messages"], stream = True)

        for chunk in stream:
            content = chunk["message"]["content"]
            full_response += content
            response_placeholder.markdown(full_response + "|")

        response_placeholder.markdown(full_response)

    st.session_state["messages"].append({"role":"assistant", "content": full_response})