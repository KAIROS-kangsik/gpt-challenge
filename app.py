from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import os
import streamlit as st
import tempfile

st.set_page_config(
    page_icon="🤖",
    page_title="DocumentGPT",
)

temp_dir = tempfile.mkdtemp()

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None


if "messages" not in st.session_state:
    st.session_state["messages"] = []

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history",
)


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_resource(show_spinner="파일 임베딩중...")
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
    embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings()

    # 수정된 부분: 임시 디렉토리에 벡터 저장
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=os.path.join(temp_dir, "vectorstore")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def check_api_key():
    try:
        return "OPENAI_API_KEY" in st.secrets
    except FileNotFoundError:
        return False


def get_api_key(key):
    st.session_state["openai_api_key"] = key


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context.
            If you don't know the answer just say you don't know.
            DON'T make anything up.
            And if the user asks a question in Korean, answer in Korean, and if the user asks a question in English, answer in English.
     
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def invoke_chain(question):
    result = chain.invoke(question)

    memory.save_context({"input": question}, {"output": result.content})
    return result


with st.sidebar:
    has_api_key = check_api_key()

    if not has_api_key:
        st.markdown(
            """
                    OpenAI API키를 입력하세요:

                    오른쪽 밑의 "Manage app"을 클릭한 후 점 세개 버튼을 클릭하세요.
                    
                    Settings를 클릭한 후 Secrets탭으로 이동하세요.

                    여기에
                    
                    OPENAI_API_KEY = ""

                    의 형태로 API키를 입력하세요. API키는 ""안에 넣어주세요.
                    """
        )

    else:
        if not st.session_state.openai_api_key:
            try:
                st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
            except:
                st.error(
                    "API 키를 찾을 수 없습니다. Streamlit Cloud의 Secrets에서 설정해주세요."
                )

        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            openai_api_key=st.session_state.openai_api_key,
            callbacks=[
                ChatCallbackHandler(),
            ],
        )
        file = st.file_uploader(
            "TXT, PDF, DOCX 파일을 업로드 하세요.",
            type=["txt", "pdf", "docx"],
        )

if st.session_state.openai_api_key:
    st.title("DocumentGPT")

    st.markdown(
        """
        안녕하세요!

        챗봇을 사용하여 파일에 대해 인공지능에게 질문하세요.

        파일을 사이드바에 업로드해 주세요.
        """
    )

    if file:
        retriever = file_embed_and_retrieve(file)
        send_message("파일을 읽었습니다. 무엇을 도와드릴까요?", "ai", save=False)
        paint_history()
        message = st.chat_input("파일에 대해 무엇이든지 물어보세요")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                    "history": load_memory,
                }
                | prompt
                | llm
            )

            with st.chat_message("ai"):
                invoke_chain(message)
    else:
        st.session_state["messages"] = []
