import streamlit as st

st.set_page_config(
    page_title = "RAG personalizado",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.header("RAG personalizado")

st.markdown("En este chat se permite al usuario agregar un fichero para que el modelo lo use.")
