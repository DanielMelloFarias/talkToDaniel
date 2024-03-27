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
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            st.error("Não foi possível ler o arquivo com as codificações comuns. Por favor, verifique a codificação do arquivo e tente novamente.")

    # Exibindo o dataframe
    st.write("Visualizando os dados:")
    st.dataframe(df)