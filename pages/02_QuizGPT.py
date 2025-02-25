from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


import os
import tempfile
import json
import streamlit as st

st.set_page_config(
    page_icon="🧐",
    page_title="QuizGPT",
)

st.title("퀴즈 GPT")

temp_dir = tempfile.mkdtemp()


if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None


function = {
    "name": "create_quiz",
    "description": "The function reads the documentation and returns a quiz with a list of questions and answers.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

try:
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=st.session_state.openai_api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a helpful assistant that is role playing as a teacher.
            
        Create 10 questions in KOREAN to test your knowledge of the text based ONLY on the following context.

        Also creates questions by determining the difficulty based on the value assigned to "Level" below.

        A Level value of “Hard” will make the question very challenging, while “Easy” will make the question easy enough for anyone with common sense.

        Each question should have 4 answers, three of them must be incorrect and one should be correct.

        Your turn!

        Context: {context}

        Level: {level}
    """,
            ),
        ]
    )

    chain = prompt | llm
except:
    pass


# @st.cache_resource(show_spinner="파일 읽는중...")
def file_embed_and_retrieve(file):

    file_content = file.read()
    file_path = os.path.join(temp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=3500,
        chunk_overlap=500,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="위키피디아 검색중...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=1, lang="ko")
    docs = retriever.get_relevant_documents(topic)
    return docs


@st.cache_resource(show_spinner="퀴즈 생성중...")
def invoke_and_create_quiz(_docs, topic, level):
    quiz = chain.invoke(
        {
            "context": _docs,
            "level": level,
        }
    ).additional_kwargs[
        "function_call"
    ]["arguments"]
    return json.loads(quiz)


with st.sidebar:
    docs = None
    if not st.session_state.openai_api_key:
        st.markdown(
            """
        ### OpenAI API 키 입력
        1. https://platform.openai.com/account/api-keys 에서 API 키를 발급받으세요.
        2. 발급받은 API 키를 아래에 입력하세요.
        3. API 키는 세션에만 저장되며, 브라우저를 닫으면 사라집니다.
        """
        )
        api_key = st.text_input("OpenAI API 키를 입력하세요", type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
            st.success("API 키가 입력되었습니다!")
            st.rerun()
        elif not st.session_state.get("openai_api_key"):
            st.warning("API 키를 입력해주세요!")
    else:
        value = st.selectbox("선택하세요", ["파일", "위키피디아"])
        level = st.selectbox("난이도를 선택하세요", ["Easy", "Hard"])
        if value == "파일":
            file = st.file_uploader(
                "TXT, PDF, DOCX 파일을 업로드 하세요.",
                type=["txt", "pdf", "docx"],
            )
            answer_toggle = st.toggle("정답보기", value=False)
            if file:
                docs = file_embed_and_retrieve(file)
                topic = file.name
        else:
            topic = st.text_input("위키피디아에서 검색할 키워드를 입력하세요")
            answer_toggle = st.toggle("정답보기", value=False)
            if topic:
                docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    QuizGPT에 오신것을 환영합니다!

    여러분이 요청한 위키피디아 기사나 업로드한 파일로 퀴즈를 만들어 지식을 테스트하고 공부하는 데 도움을 드립니다.

    사이드바에서 파일을 업로드하거나 위키백과에서 검색하여 시작하세요.
    """
    )
else:
    quiz_object = invoke_and_create_quiz(docs, topic, level)
    with st.form("question form"):
        st.session_state["count"] = 0
        for question in quiz_object["questions"]:
            value = st.radio(
                question["question"],
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if answer_toggle:
                for elem in question["answers"]:
                    if elem["correct"] == True:
                        st.info(f"💡 정답:  {elem['answer']}")
                        break

            if value:
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("정답입니다.")
                    st.session_state["count"] += 1
                else:
                    st.error("틀렸습니다.")
        st.form_submit_button("제출")
    st.write(f"점수: {st.session_state['count']} / {len(quiz_object['questions'])}")
    if st.session_state["count"] == len(quiz_object["questions"]):
        st.balloons()
