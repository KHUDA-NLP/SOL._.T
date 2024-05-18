from dotenv import load_dotenv
load_dotenv() # dotenv 파일에 있는 API KEY 불러오기

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate

query = input()

# model 불러오기
chat_model = ChatOpenAI(model_name='gpt-3.5-turbo', 
                        streaming=True, callbacks=[StreamingStdOutCallbackHandler()], 
                        temperature = 1)

template = """
너는 연애에 대해 조언해 줄 수 있는 상담사야. 
내가 연인과 처한 문제 상황을 듣고, 연인에게 공감하여 해결책을 제시해 줘.
내가 처한 문제 상황은 다음과 같아.

{situation}
"""

# ChatGPT에게 역할 부여
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 사용자가 입력할 매개변수 template 선언
human_template = "{situation}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# chat prompt template에 전달
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt.format_messages(situation=query)

# chain을 통해 참고 문서와 사용자 질의 묶기
qa = RetrievalQA.from_chain_type(
    llm = chat_model,
    chain_type_kwargs={"prompt":chat_prompt},
    chain_type = "stuff",
    #retriever = docsearch.as_retriever (이건크로마 객체 받은 후 도큐먼트불러올수잇음)
)

result = qa()

# qa에 상황 입력 받아서 넘기는 거 해야 함!!

