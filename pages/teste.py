import streamlit as st
import pandas as pd

def main():
    # Título da página/aplicativo
    st.title("Visualizador de CSV com Streamlit")

    # Carregamento do arquivo
    uploaded_file = st.file_uploader("Carregue seu arquivo CSV aqui", type=["csv"])

    # Verifica se um arquivo foi carregado
    if uploaded_file is not None:
        # Lendo o arquivo CSV
        df = pd.read_csv(uploaded_file)

        # Exibindo o dataframe
        st.write("Visualizando os dados:")
        st.dataframe(df)

# Executa a função principal
if __name__ == "__main__":
    main()