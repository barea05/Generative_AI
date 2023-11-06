## Conversational QandA chatbot

import streamlit as st

from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain.chat_models import ChatOpenAI


##stramlit UI
st.set_page_config(page_title='Conversational Q&A Chatbot')
st.header("Hey, Let's Chat")
from dotenv import load_dotenv
load_dotenv()

import os
openai_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(temperature=0.5)
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[
        SystemMessage(content='You are comedian AI assistant')
    ]


def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

input = st.text_input("Input: " , key = "Input")
response = get_chatmodel_response(input)
submit = st.button("Ask the question")

##if ask button is called

if submit:
    st.subheader('The response is')
    st.write(response)