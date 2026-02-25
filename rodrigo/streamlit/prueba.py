import streamlit as st
import ollama
import pandas as pd
import numpy as np

st.set_page_config(
    page_title = "YodaChat",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.header("YodaChat")

with st.sidebar:
    st.subheader("Sidebar")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append(
        {"role":    "system",
         "content": """Estas interpretando al personaje Yoda de la saga de peliculas Star Wars. Este se caracteriza por 
                    ser un personaje de gran sabiduria y por hablar "al revés"; basicámente usa con frecuencia un recurso 
                    literario llamado "hipérbaton". Debes comportarte como él, responde como lo haría él."""}
    )

with st.container():
    for message in st.session_state["messages"]:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

prompt = st.chat_input("Introduzca el texto a traducir")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state["messages"].append({"role":"user", "content":prompt})

    response = ollama.chat(model = "llama3.2", messages = st.session_state["messages"])
    response_content = response["message"]["content"].strip('"')

    with st.chat_message("assistant"):
        st.markdown(response_content)
    st.session_state["messages"].append({"role":"assistant", "content": response_content})