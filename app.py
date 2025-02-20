from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
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

    # 수정된 부분: 임시 디렉토리에 벡터 저장
    vectorstore = FAISS.from_documents(
        docs,
        embeddings,
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
        file = st.file_uploader(
            "TXT, PDF, DOCX 파일을 업로드 하세요.",
            type=["txt", "pdf", "docx"],
        )

if st.session_state.openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        openai_api_key=st.session_state.openai_api_key,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

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
