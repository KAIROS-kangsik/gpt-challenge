from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

import streamlit as st

st.set_page_config(
    page_icon="📜",
    page_title="SiteGPT",
)

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if "messages_memory" not in st.session_state:
    st.session_state["messages_memory"] = []


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


try:
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-turbo-preview",
        openai_api_key=st.session_state.openai_api_key,
    )

    choose_llm = ChatOpenAI(
        temperature=0,
        streaming=True,
        openai_api_key=st.session_state.openai_api_key,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )
except:
    pass


give_score_prompt = ChatPromptTemplate.from_template(
    """
    다음 'Context'에 있는 정보로만 사용자의 질문에 답하세요. 'Context'에 없는 정보는 절대 답변에 포함시키지 마세요.

    만약 'Context'에 질문에 대한 답이 없다면, 반드시 '모르겠습니다.'라고만 답하고, 임의의 값을 지어내지 마세요.

    답변 후, 답변의 정확성을 0~5점으로 평가하세요.  
    - 5: 'Context'에 기반한 완전히 정확한 답변  
    - 0: 'Context'에 답이 없거나 틀린 경우  

    0점인 경우에도 답변의 점수를 '반드시' 포함시키세요.

    Context: {context}

    Examples:

    Question: 달은 얼마나 멀리 있나요?
    Answer: 달은 384,400km 떨어져 있습니다.
    Score: 5

    Question: 태양은 얼마나 멀리 있나요?
    Answer: 모르겠습니다.
    Score: 0

    당신 차례예요!

    Question: {question}
"""
)


def ask_each_doc(inputs):
    docs = inputs["context"]
    question = inputs["question"]
    chain = give_score_prompt | llm
    answer_array = []
    for doc in docs:
        result = chain.invoke(
            {
                "context": doc.page_content,
                "question": question,
            }
        )
        answer_array.append(result)
    result_object = {
        "question": question,
        "answers": answer_array,
    }
    # st.write(docs, question)
    return result_object


def save_message(message, role):
    st.session_state.get("messages_memory").append(
        {
            "content": message,
            "role": role,
        }
    )


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 아래의 answers 리스트에서 질문과 가장 관련이 깊고, 동시에 가장 높은 점수를 가진 답변을 선택해야 합니다.

선택한 답변의 텍스트를 그대로 출력해 주세요. 당신의 출력이 바로 최종 답변이 되므로, 어떤 추가적인 텍스트, 접두사, 점수 등을 포함하지 마세요. 오직 선택한 답변의 원문 자체만 출력해야 합니다.

예를 들어, 만약 선택한 답변이 '한국의 수도는 서울입니다.'라면, 출력은 다음과 같아야 합니다:

한국의 수도는 서울입니다.

주의:
- 'Answer: ', '답변: ' 등의 접두사를 붙이지 마세요.
- 답변을 따옴표로 감싸지 마세요.
- 점수나 다른 설명을 추가하지 마세요.
- 선택한 답변의 텍스트를 정확히 그대로 출력해 주세요.

항상 한글로만 대답해 주세요.

Answers list: {answers}
""",
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]
    chain = choose_prompt | choose_llm

    answers = "\n\n".join(answer.content for answer in answers)
    result = chain.invoke(
        {
            "answers": answers,
            "question": question,
        }
    )
    return result


def page_parse(soup):
    astro_fxeopwe4 = soup.find("div", class_="astro-fxeopwe4")
    if astro_fxeopwe4:
        astro_fxeopwe4.decompose()
    card_grid = soup.find("div", class_="card-grid")
    if card_grid:
        card_grid.decompose()
    astro_breadcrumbs = soup.find("astro-breadcrumbs")
    if astro_breadcrumbs:
        astro_breadcrumbs.decompose()
    button = soup.find("div", class_="!mt-0 self-center")
    if button:
        button.decompose()

    select_overview_html = soup.find("main", class_="astro-bguv2lll")

    if select_overview_html:
        title = select_overview_html.find("h1", id="_top")
        title_text = title.get_text(strip=True) if title else "No title found"

        content = select_overview_html.find(
            "div", class_="sl-markdown-content astro-cedpceuv"
        )
        content_text = content.get_text(strip=True) if content else "No content found"

        text = f"Title: {title_text}\n\ncontent: {content_text}"
        return text
    return ""


@st.cache_resource(show_spinner="사이트 읽는중...")
def split_and_retrieve(url):
    loader = SitemapLoader(
        url,
        filter_urls=[
            "https://developers.cloudflare.com/ai-gateway/*",
            "https://developers.cloudflare.com/vectorize/*",
            "https://developers.cloudflare.com/workers-ai/*",
        ],
        parsing_function=page_parse,
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    # st.write(docs)
    return vector_store.as_retriever()
    # return docs


def paint_messages():
    if len(st.session_state.get("messages_memory")):
        for obj in st.session_state.get("messages_memory"):
            with st.chat_message(obj["role"]):
                st.markdown(obj["content"])


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
        st.success("API입력 성공!")
        st.write("Cloudflare 공식문서를 위한 SiteGPT에 오신것을 환영합니다.")

if st.session_state["openai_api_key"]:
    st.markdown(
        """
    # SiteGPT

    웹사이트 Cloudflare의 내용에 대해 질문하세요.
"""
    )
    url = "https://developers.cloudflare.com/sitemap-0.xml"
    retrieve = split_and_retrieve(url)
    # with st.chat_message("human"):
    query = st.chat_input()

    if query:
        save_message(query, "human")
        paint_messages()

        chain = (
            {
                "context": retrieve,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(ask_each_doc)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            chain.invoke(query)

        st.rerun()

if len(st.session_state.get("messages_memory")):
    if st.session_state.get("messages_memory")[-1]["role"] == "ai":
        paint_messages()
