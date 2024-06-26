import streamlit as st
import pandas as pd

# Título da página/aplicativo
st.title("Visualizador de CSV com Streamlit")

# Carregamento do arquivo
uploaded_file = st.file_uploader("Carregue seu arquivo CSV aqui", type=["csv"])

# Verifica se um arquivo foi carregado
if uploaded_file is not None:
        # Lendo o arquivo CSV
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', delimiter=";")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

    # Exibindo o dataframe
    st.write("Visualizando os dados:")
    st.dataframe(df)