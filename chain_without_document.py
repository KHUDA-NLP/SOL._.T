# document를 받기 전 prompt engineering만 진행

from dotenv import load_dotenv
load_dotenv()  # dotenv 파일에서 API KEY 불러오기

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

# 사용자 입력을 받습니다.
query = input("상황을 입력하세요: ")

# 모델 불러오기
chat_model = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=1
)

# 템플릿 정의
template = """
너는 연애에 대해 조언해 줄 수 있는 상담사야. 
내가 연인과 처한 문제 상황을 듣고, 연인에게 공감하여 해결책을 제시해 줘.
내가 처한 문제 상황은 다음과 같아.

{situation}
"""

# 시스템 메시지 프롬프트 생성
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 사용자 메시지 프롬프트 생성
human_template = "{situation}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# ChatPromptTemplate 생성
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# LLMChain 생성
llm_chain = LLMChain(
    llm=chat_model,
    prompt=chat_prompt
)

# 질의에 대한 응답 생성
result = llm_chain.run(situation=query)

# 결과 출력
print(result)