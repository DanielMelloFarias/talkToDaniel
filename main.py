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
api_video = st.secrets["token_video"]


# Função para gerar vídeos baseados em prompts de texto e URLs de avatar, com suporte para diferentes vozes baseadas no gênero.
def generate_video(prompt, avatar_url, gender):
    url = "https://api.d-id.com/talks"
    headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization" : "Basic c2Vydmlkb3Jmb3JleEBnbWFpbC5jb20:HWkojUOp0cG85IIVt5PDy"
}
    if gender == "Female":
        payload = {
            "script": {
                "type": "text",
                "subtitles": "false",
                "provider": {
                    "type": "microsoft",
                    "voice_id": "pt-BR-BrendaNeural"
                },
                "ssml": "false",
                "input":prompt
            },
            "config": {
                "fluent": "false",
                "pad_audio": "0.0"
            },
            "source_url": avatar_url
        }

    if gender == "Male":
        payload = {
            "script": {
                "type": "text",
                "subtitles": "false",
                "provider": {
                    "type": "microsoft",
                    "voice_id": "pt-BR-AntonioNeural"
                },
                "ssml": "false",
                "input":prompt
            },
            "config": {
                "fluent": "false",
                "pad_audio": "0.0"
            },
            "source_url": avatar_url
        }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 201:
            print(response.text)
            res = response.json()
            id = res["id"]
            status = "created"
            while status == "created":
                getresponse =  requests.get(f"{url}/{id}", headers=headers)
                print(getresponse)
                if getresponse.status_code == 200:
                    status = res["status"]
                    res = getresponse.json()
                    print(res)
                    if res["status"] == "done":
                        video_url =  res["result_url"]
                    else:
                        time.sleep(10)
                else:
                    status = "error"
                    video_url = "error"
        else:
            video_url = "error"   
    except Exception as e:
        print(e)      
        video_url = "error"      
        
    return video_url


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

contexto = "Voce está numa entrevista para uma vaga de Especialista em Inovação e IA para empresa: Magnum Tires. Agora esse é um texto da minha vida e currículo. Com uma jornada profissional marcada por inovação, liderança e um profundo compromisso com a transformação digital na saúde, eu, Daniel Gomes de Mello Farias, trilhei um caminho que mescla conhecimento técnico avançado com uma visão estratégica voltada para o futuro da tecnologia aplicada à saúde. Mestre em Modelagem Computacional do Conhecimento e Ciência e Tecnologia na Saúde, minha formação acadêmica e profissional reflete uma dedicação contínua ao aprimoramento e aplicação prática de soluções inovadoras. Minha experiência como Líder Técnico no Hospital Sírio Libanês e PMO de Projetos Estratégicos na Secretaria de Saúde do Estado de Alagoas destaca minha capacidade de conduzir equipes e projetos com o objetivo de integrar tecnologias emergentes como a Inteligência Artificial, Realidade Aumentada/Virtual, e Ciência de Dados, para transformar a experiência de pacientes e profissionais de saúde. Essa trajetória é complementada por minha atuação como mentor e avaliador de startups no programa Inovativa Brasil, onde auxilio na orientação de jovens empresas que buscam revolucionar o mercado de saúde com tecnologias disruptivas. As certificações profissionais em Gestão de Projetos e Inteligência Artificial e Machine Learning atestam meu compromisso com a excelência e a constante atualização em áreas chave para a inovação na saúde. Além disso, a honra de ter sido reconhecido com o 3º melhor paper no XI North Northeast Congress of Research and Innovation sublinha minha contribuição acadêmica e profissional para o campo da engenharia e tecnologia. Minha visão é conduzida pela crença de que a verdadeira inovação surge na interseção entre a tecnologia, a saúde e a capacidade de antecipar as necessidades futuras da sociedade. Por isso, me dedico a explorar como a análise de dados e as tecnologias emergentes podem ser aplicadas para prever tendências de saúde, otimizar tratamentos e promover um sistema de saúde mais integrado e eficiente. Em busca de novos desafios e oportunidades, estou sempre aberto a colaborações que permitam utilizar minha expertise técnica, liderança e paixão por inovação, para criar soluções que não apenas resolvam problemas complexos, mas que também melhorem a qualidade de vida das pessoas. Através da minha jornada, pretendo continuar impactando positivamente o setor da saúde, liderando pelo exemplo e inspirando outros a explorarem o potencial ilimitado da tecnologia para fazer a diferença no mundo. Você está sendo entrevistado então responda sempre de maneira educada as perguntas, seja gentil e responda sempre na 1 pessoa. Ao final da resposta (baseado no texto anterior) faça uma outra pergunta pra pessoa (mas n mencione saúde). Responda smepre em até 1 parágrafo de até 50 palavras. Caso for falar da empresa Magnum Tires, pode ser criativo e pegar da sua base."
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
            # Atualiza o histórico de chat no estado da sessão para incluir a interação mais recente.
            try:
                perguntaVideo = model.generate_content(['Extraia APENAS o texto da pergunta, sem mais nada', full_response])
                print ("Pergunta: ", perguntaVideo.text)
                video_url = generate_video(perguntaVideo.text, "https://iili.io/JjFrGXR.jpg", "Male")  # Call your video generation function here
                if video_url!= "error":
                    st.text("Vídeo Gerado!")
                    # Placeholder for displaying generated video
                    #st.subheader("Vídeo Gerado")
                    st.video(video_url)  # Replace with the actual path
                else:
                    st.text("Sorry... Try again")
            except Exception as e:
                print(e)
                st.text("Sorry... Try again")
                st.session_state.messages.append(video_url)
            st.session_state.history = chat.history
            st.session_state.history = []