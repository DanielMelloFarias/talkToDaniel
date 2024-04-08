# Tutorial: https://www.youtube.com/watch?v=GztkOuIku04

import streamlit as st
from streamlit_chat import message

import google.generativeai as genai

import time
import requests
import random

#api_key = os.environ.get('KeyMaster')
#api_video = os.environ.get('token_video')

# Estas são as chaves API armazenadas como segredos no Streamlit, usadas para autenticação em serviços externos.
api_key = st.secrets["KeyMaster"]


# Verifica e inicializa o estado da sessão para o histórico de chat, se necessário.
if "history" not in st.session_state:
    st.session_state.history = []

# Configura a chave API para a genai e inicia a configuração do modelo de chat.
genai.configure(api_key = api_key)
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history = st.session_state.history)

# Configurações iniciais da página Streamlit, como título e ícone.
st.set_page_config(
    page_title="Talk With Daniel Mello",
    page_icon="🔥"
)
st.title("Talk With Daniel Mello")
st.caption("A Chatbot To Talk with Daniel Mello ")


app_key = api_key

if "app_key" not in st.session_state:
    app_key = api_key
    if app_key:
        st.session_state.app_key = api_key

if "history" not in st.session_state:
    st.session_state.history = []


genai.configure(api_key = api_key)
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history = st.session_state.history)


# Sidebar com botão para limpar a janela de chat.
with st.sidebar:
    if st.button("Clear Chat Window", use_container_width=True, type="primary"):
        st.session_state.history = []
        st.rerun()

# Exibe as mensagens de chat anteriores.
for message in chat.history:
    role ="assistant" if message.role == 'model' else message.role
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

contexto = "Voceé especialista em inovação... "
# Permite ao usuário enviar uma nova mensagem e exibe a resposta do assistente
if "app_key" in st.session_state:
    if prompt := st.chat_input(""):
        prompt = prompt.replace('\n', ' \n')
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                full_response = ""
                for chunk in chat.send_message(contexto + prompt, stream=True):
                    word_count = 0
                    random_int = random.randint(5,10)
                    for word in chunk.text:
                        full_response+=word
                        word_count+=1
                        if word_count == random_int:
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "_")
                            word_count = 0
                            random_int = random.randint(5,10)
                message_placeholder.markdown(full_response)
            except genai.types.generation_types.BlockedPromptException as e:
                st.exception(e)
            except Exception as e:
                st.exception(e)
            st.session_state.history = chat.history
            st.session_state.history = []