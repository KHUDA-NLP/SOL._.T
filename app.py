import streamlit as st
from utils import print_messages
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate # MessagePlaceHolder
from langchain_core.output_parsers import StrOutputParser
import os

from dotenv import load_dotenv

st.set_page_config(
    page_title = "연애 솔루션 챗봇 Sol-T",
    page_icon = "SOL-T💞")
st.title("연애 솔루션 챗봇 SOL-T💞")

# API key 설정
load_dotenv()
os.environ.get("OPENAI_API_KEY")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 기록 출력해주는 함수. 
print_messages()

if user_input := st.chat_input("어떤 것이 궁금하신가요?"):
    # 사용자 입력 
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role = "user", content = user_input))

    # LLM 답변 생성
    prompt = ChatPromptTemplate.from_template(
        """질문에 대해 간결하지만 최대한 친절하게 답변하라.     
{question}
""")

    chain = prompt | ChatOpenAI() | StrOutputParser()
    msg = chain.invoke({"question" : user_input})

    # AI 답변   
    with st.chat_message("assistant"):
        # msg = f" '아!! {user_input}' 이렇게 답변하셨군요"
        st.write(msg)
        st.session_state["messages"].append(ChatMessage(role = "assistant", content = msg))
