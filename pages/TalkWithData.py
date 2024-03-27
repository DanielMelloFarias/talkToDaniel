#Fonte: https://github.com/saha-trideep/AI-Data-Assistant/tree/main

import streamlit as st

import re
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI



from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentType, initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper


#-----------------------------------------------------------#

# Function to load and set API key
def load_api_key():
    api_key = st.secrets["KeyMaster"]
    return api_key
    
# Function to display welcome message and sidebar
def display_welcome():
    st.title("Sou DanielData, O assistente para Análise de Dados🤖")
    st.write('Olá!!👋 Sou seu assistente de dados e estou aqui para lhe ajudar com projetos de Ciência de Dados. 💚')
    
    # side bar 
    with st.sidebar:
        st.write('*Estou aqui para ajudar na jornada dos dados. Para começar, preciso de um arquivo .CSV* ')
        st.caption('''**Nossa jornada começa com um arquivo CSV...
                Vamos mergulhar na compreensão dos dados e...
                explorar todas as ideias e soluções.. **
                ''')
        # divider
        st.divider()
        st.caption('Daniel Mello 💛', unsafe_allow_html=True)
 
# function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True 

# Function to handle user file upload
def handle_file_upload():
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
        return df
    return None

#-----------------------------------------------------------#

# Function to handle suggestions 
def suggestion_model(api_key, topic):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)    
    data_science_prompt = PromptTemplate.from_template("You are a genius data scientist. Write me a solution {topic}. Escreva em pt-br")
    prompt_chain = LLMChain(llm=llm, prompt=data_science_prompt, verbose=True)
    resp = prompt_chain.run(topic)
    return resp

# # Function to load Wikipedia research based on the prompt
@st.cache_resource
def wiki(prompt):
    wiki_research = WikipediaAPIWrapper().run(prompt)
    return "Wikipedia Research for " + prompt

# Function to handle problem template chain
def prompt_templates():
    data_problem_template = PromptTemplate(
    input_variables=['business_problem'],
    template='Convert the following business problem into a data science problem: {business_problem}. Responda em pt-br'
    )
    template='''Give a list of machine learning algorithms and as well as step by step 
    python code for any one algorithm that you think is suitable to solve 
    this problem: {data_problem}, while using this Wikipedia research: {wikipedia_research}. Responda em pt-br'''
    
    model_selection_template = PromptTemplate(
        input_variables=['data_problem', 'wikipedia_research'],
        template=template
    )

    return data_problem_template, model_selection_template

# Define the cache_data decorator for chains
@st.cache_data
def chains(_model):
    
    data_problem_chain = LLMChain(llm=_model, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
    model_selection_chain = LLMChain(llm=_model, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
    sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
    return sequential_chain

# Define the cache_data decorator for chains output
@st.cache_data
def chains_output(prompt, wiki_research, _model):
    my_chain = chains(_model)
    my_chain_output = my_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
    my_data_problem = my_chain_output["data_problem"]
    my_model_selection = my_chain_output["model_selection"]
    return my_data_problem, my_model_selection

# Function to extract machine learning algorithms from the output
# @st.cache_data
# def list_to_selectbox(input_text):
#     algorithms_list = []
#     lines = input_text.split('\n')

#     for line in lines:
#         # Use regular expression to find lines that seem to contain algorithm names
#         match = re.search(r'\b([A-Za-z\s]+)\b', line)
#         if match:
#             algorithm_name = match.group(1).strip()
#             algorithms_list.append(algorithm_name)

#     # Insert "Select Algorithm" at the beginning
#     algorithms_list.insert(0, "Select Algorithm")

#     return algorithms_list

# Function is part of the LangChain library and is used to create a Python Agent
# @st.cache_resource
# def python_agent(_model):
#     agent_executor = create_python_agent(
#         llm=_model,
#         tool=PythonREPLTool(),
#         verbose=True,
#         agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#         handle_parsing_errors=True,
#         )
#     return agent_executor

# @st.cache_data
# def python_solution(my_data_problem, selected_algorithm, user_csv, _model):
#     solution = python_agent(_model).run(
#         f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}."
#     )
#     return solution
#----------------------------------------------------------#

# Function to diplay a overview of data
@st.cache_data(experimental_allow_widgets=True)
def data_overview(df, _pandas_agent):
    st.write("**Resumo dos dados**")
    st.write("As primeiras linhas dos dados...")
    st.write(df.head())
    
    columns_df = _pandas_agent.run("What are the meaning of the columns? Escreva em pt-br")
    if columns_df is not None:
        st.write(columns_df)
    else:
        st.warning("Unable to retrieve column information.")
    
    st.write("**Valores 'Missing'**")
    st.write("Número de dados 'missing' em cada coluna:")
    st.write(df.isnull().sum())
    
    st.write("**Valores Replicados (duplicados)**")
    duplicates = _pandas_agent.run("Are there any duplicate values and if so where?.Responda em pt-br")
    st.write(duplicates)
    
    st.write("**Sumário dos Dados **")
    st.write(df.describe())
    
    # Shape of the Dataset
    st.write("**Shape odo DataSet**")
    st.write(f"Os dados tem: {df.shape[0]} linhas (rows) and {df.shape[1]} colunas (columns).")
    
    # Skewness for Numeric Variables
    st.write("**Assimetria das Variáveis **")
    numeric_columns = df.select_dtypes(include='number').columns
    for col in numeric_columns:
        skewness = df[col].skew()
        st.write(f"Assimetria : '{col}': {skewness}")
    
    # # Data Types
    # st.write("**Data Types**")
    # selected_column_datatypes = st.selectbox("Select a column for data types:", df.columns)
    # if selected_column_datatypes:
    #     data_type = _pandas_agent.run(f"df['{selected_column_datatypes}'].dtypes")
    #     st.write(f"Data type of '{selected_column_datatypes}': {data_type}")
    # else:
    #     st.warning("Select a column to display its data type.")
    
    # # Visualizations
    # st.header("Visualizations")

    # # Histogram
    # st.subheader("Histogram")
    # selected_numeric_column = st.selectbox("Select a numeric column for histogram:", df.select_dtypes(include='number').columns)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.histplot(df[selected_numeric_column], ax=ax, color='#4CAF50')
    # st.pyplot(fig)

    # # Box Plot
    # st.subheader("Box Plot")
    # selected_numeric_column_box = st.selectbox("Select a numeric column for box plot:", df.select_dtypes(include='number').columns)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.boxplot(x=df[selected_numeric_column_box], color='#2196F3')
    # st.pyplot(fig)

    # # Pair Plot
    # st.subheader("Pair Plot")
    # selected_numeric_columns_pair = st.multiselect("Select numeric columns for pair plot:", df.select_dtypes(include='number').columns)
    # if selected_numeric_columns_pair:
    #     pair_plot = sns.pairplot(df[selected_numeric_columns_pair])
    #     st.pyplot(pair_plot.fig)
    # else:
    #     st.warning("Select at least two numeric columns for pair plot.")

    return  

# Function to perform user specific task
@st.cache_data(experimental_allow_widgets=True)
def perform_pandas_task(task, _pandas_agent):
    if task:
        return _pandas_agent.run(task)
    else:
        return f"Task '{task}' not recognized."

# Function to handle query
@st.cache_data(experimental_allow_widgets=True)
def perform_eda(input, _pandas_agent):
    if input:
        result = perform_pandas_task(input, _pandas_agent)
        st.write(f"**Result of '{input}'**")
        st.write(result)

# Function to display a brief information about a variable in a dataset. 
@st.cache_data
def variable_info(df, var, varX):
    # Summary Statistics
    st.write(f"Resumo Estatístico: '{var}':")
    st.write(df[var].describe())
    
    # line plot
    st.line_chart(df, x="Produto", y=[var])

    # Distribution Visualization
    st.write(f"Distribuição: '{var}':")
    fig, ax = plt.subplots()
    sns.histplot(df[var], kde=True, ax=ax, color='#4CAF50')
    st.pyplot(fig)

    # Box Plot
    st.write(f"BoxPlot: '{var}':")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[var], ax=ax, color='#4CAF50')
    st.pyplot(fig)

    # Value Counts for Categorical Variables
    if df[var].dtype == 'O':  # Check if the variable is categorical
        st.write(f"Contagem de dados: '{var}':")
        st.write(df[var].value_counts())
    else:
        st.write(f"A detecção de valores discrepantes e os testes de normalidade não são aplicáveis para variáveis  '{var}' .")

    # Outliers Detection
    st.write(f"Detecção de Outliers para a varíavel:  '{var}':")
    if df[var].dtype != 'O':  # Check if the variable is not categorical
        z_scores = stats.zscore(df[var])
        outliers = df[(z_scores > 3) | (z_scores < -3)][var]
        st.write(outliers)
    else:
        st.write("A detecção de valores discrepantes não é aplicável a variáveis categóricas")

    # Normality Test
    st.write(f"Teste normal para variável: '{var}':")
    if df[var].dtype != 'O':  # Check if the variable is not categorical
        _, p_value = stats.normaltest(df[var].dropna())
        st.write(f"P-value: {p_value}")
        if p_value < 0.05:
            st.write("A variável não segue uma distribuição normal.")
        else:
            st.write("A variável segue uma distribuição normal.")
    else:
        st.write("O teste de normalidade não é aplicável para variáveis categóricas.")

    # Missing Values
    st.write(f"Valores 'missing': '{var}':")
    st.write(df[var].isnull().sum())

    # Data Type
    st.write(f"Tipo (type) dos dados: '{var}':")
    st.write(df[var].dtype)
    return 

 
GOOGLE_API_KEY=load_api_key()
display_welcome()

# initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}
st.button("Click Aqui para Começar..", on_click=clicked, args=[1])
if st.session_state.clicked[1]: 
    user_csv = handle_file_upload()
    if user_csv is not None:
        # Initialize pandas_agent
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
        pandas_agent = create_pandas_dataframe_agent(llm, user_csv, verbose=True, handle_parsing_errors=True)

        st.header("Explorando a Análise de dados")
        st.subheader("Informações Gerais sobre o Dataset")
        data_overview(user_csv, pandas_agent)
    
        st.subheader("Variável de Estudo:")
            
        user_question_variable = st.selectbox("Qual variável / feature é importante??", user_csv.select_dtypes(include='number').columns)
        #user_question_variable_X = st.selectbox("Qual variável / feature é importante??", user_csv.columns)
        if user_question_variable:
            variable_info(user_csv, user_question_variable, user_question_variable_X)
            st.subheader("Estudo Aprofundado:")
            task_input = st.text_input("O que mais vc gostaria de analisar?")
            if task_input:
                st.write(perform_pandas_task(task_input, pandas_agent))

                st.divider()
                st.header("Problema de Ciências de Dados - Após Analisar os dados")
                st.write("""Agora que temos uma compreensão sólida dos dados em mãos 
                            e uma compreensão clara da variável que pretendemos investigar,
                            é importante reformularmos nosso problema de negócios
                            em um problema de ciência de dados.""")
                
                # Get user input
                prompt = st.text_area('Qual problema de negócio vc qr resolver?')

                # Display results on button click
                if st.button("Sugestões?"):
                    if prompt:
                        wiki_research = wiki(prompt)
                        my_data_problem, my_model_selection = chains_output(prompt, wiki_research, llm)
                        st.write("**Problema de Ciência de Dados:**")
                        st.write(my_data_problem)
                        st.write("**Sugestões de Algoritmos de ML:**")
                        st.write(my_model_selection)
                        # algorithm_list = list_to_selectbox(my_model_selection)
                        # st.write(algorithm_list)
                        # selected_algorithm = st.selectbox("Select Machine Learning Algorithm", algorithm_list)
                        
                        # if selected_algorithm != "Select Algorithm":
                        #     st.subheader("Assumption")
                        #     solution = python_solution(my_data_problem, selected_algorithm, user_csv, pandas_agent)
                        #     st.write(solution)
                            

        with st.sidebar:
            with st.expander("Quais são as etapas da EDA"):
                topic = 'Quais são as etapas da Análise Exploratória de Dados'
                resp = suggestion_model(GOOGLE_API_KEY, topic)
                st.write(resp)

            with st.expander("Get Help"):
                llm_suggestion = st.text_area("Me faça perguntas sobre ciência de dados:")

                if st.button("Tell me"):
                    llm_result = suggestion_model(GOOGLE_API_KEY, llm_suggestion)
                    st.write(llm_result)