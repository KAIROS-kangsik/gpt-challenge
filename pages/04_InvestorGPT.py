import streamlit as st
import os
import openai as client
import json
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(
    page_icon="📈",
    page_title="InvestorGPT",
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None

if "run_id" not in st.session_state:
    st.session_state.run_id = None


# 함수들 정의
def get_website_url(inputs):
    query = inputs["query"]
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_cse_id = os.environ.get("GOOGLE_CSE_ID")

    if not google_api_key or not google_cse_id:
        return "Google API 키와 CSE ID가 설정되지 않았습니다."

    url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cse_id}&q={query}"

    response = requests.get(url)
    results = response.json()

    if "items" not in results:
        return "검색 결과가 없습니다."

    # 검색 결과 중 상위 5개만 반환
    websites = [item["link"] for item in results["items"][:5]]
    return json.dumps(websites)


def get_website_content(inputs):
    url = inputs["url"]
    try:
        response = requests.get(url)
        html_content = response.text

        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(html_content, "html.parser")

        # 스크립트, 스타일 태그 제거
        for script in soup(["script", "style"]):
            script.extract()

        # 텍스트 추출
        text = soup.get_text(separator="\n", strip=True)

        # 길이 제한 (API 제한을 고려)
        if len(text) > 10000:
            text = text[:10000] + "..."

        return text
    except Exception as e:
        return f"오류 발생: {str(e)}"


def search_wikipedia(inputs):
    topic = inputs["topic"]
    try:
        response = requests.get(
            f"https://ko.wikipedia.org/api/rest_v1/page/summary/{topic}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = response.json()

        if "extract" in data:
            return data["extract"]
        else:
            return "위키피디아에서 정보를 찾을 수 없습니다."
    except Exception as e:
        return f"위키피디아 검색 오류: {str(e)}"


# 어시스턴트 함수 정의
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_website_url",
            "description": "주어진 질문에 대한 관련 웹사이트 URL을 검색합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 질문 또는 키워드",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_website_content",
            "description": "주어진 URL에서 웹사이트 내용을 가져옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "내용을 가져올 웹사이트 URL",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "위키피디아에서 정보를 검색합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "위키피디아에서 검색할 주제",
                    },
                },
                "required": ["topic"],
            },
        },
    },
]


# 어시스턴트 생성 함수
def create_assistant(api_key):
    try:
        client.api_key = api_key

        # 어시스턴트 생성
        assistant = client.beta.assistants.create(
            name="Research Agent",
            instructions="""
            당신은 연구 에이전트입니다.
            
            주제에 대한 정보를 찾아서 제공합니다.
            
            항상 한글로 답변을 제공합니다.
            
            먼저 get_website_url을 이용하여 웹사이트들을 먼저 찾아서 그 웹사이트에서 정보를 찾으세요.
            
            그런다음 웹사이트의 정보가 미흡하다면 search_wikipedia를 이용하여 위키피디아에서 정보를 찾으세요.
            
            만약 위키피디아에서 정보를 찾았다면 "웹사이트에서 정보를 찾지못해 위키피디아에서 찾았습니다."라는 문구를 최종 답변에 포함시켜 주세요.
            """,
            model="gpt-4o",
            tools=functions,
        )

        # 쓰레드 생성
        thread = client.beta.threads.create()

        st.write(f"어시스턴트 생성 완료: {assistant.id}")
        st.write(f"쓰레드 생성 완료: {thread.id}")

        return assistant.id, thread.id
    except Exception as e:
        st.error(f"어시스턴트 생성 중 오류: {str(e)}")
        return None, None


# 메시지 전송 및 처리 함수
def send_message(thread_id, content, api_key):
    client.api_key = api_key
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


# 실행 상태 확인 함수
def check_run_status(thread_id, run_id, api_key):
    client.api_key = api_key
    return client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )


# 메시지 가져오기 함수
def get_messages(thread_id, api_key):
    client.api_key = api_key
    return client.beta.threads.messages.list(thread_id=thread_id)


# 도구 호출 처리 함수
def handle_tool_calls(thread_id, run_id, tool_calls, api_key):
    client.api_key = api_key
    tool_outputs = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        if function_name == "get_website_url":
            output = get_website_url(function_args)
        elif function_name == "get_website_content":
            output = get_website_content(function_args)
        elif function_name == "search_wikipedia":
            output = search_wikipedia(function_args)
        else:
            output = f"함수 '{function_name}'은 지원되지 않습니다."

        tool_outputs.append(
            {
                "tool_call_id": tool_call.id,
                "output": output,
            }
        )

    # 도구 호출 결과 제출
    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_outputs,
    )


# 사이드바 설정
with st.sidebar:
    st.title("ResearchGPT")

    # GitHub 링크 추가
    st.markdown("[GitHub 저장소](https://github.com/KAIROS-kangsik/gpt-challenge)")

    # API 키 입력
    if "openai_api_key" not in st.session_state:
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
            # API 키 저장
            st.session_state.openai_api_key = api_key

            # 어시스턴트 및 쓰레드 생성
            assistant_id, thread_id = create_assistant(api_key)

            if assistant_id and thread_id:
                st.session_state.assistant_id = assistant_id
                st.session_state.thread_id = thread_id
                st.success(
                    f"API 키가 입력되었습니다! 어시스턴트 ID: {assistant_id}, 쓰레드 ID: {thread_id}"
                )
                st.rerun()
            else:
                st.error("어시스턴트 또는 쓰레드를 생성하지 못했습니다.")
        elif not st.session_state.get("openai_api_key"):
            st.warning("API 키를 입력해주세요!")
    else:
        st.success("API 연결 성공!")
        st.write(f"어시스턴트 ID: {st.session_state.assistant_id}")
        st.write(f"쓰레드 ID: {st.session_state.thread_id}")
        st.write("연구 주제에 대해 질문하세요.")

        # 재설정 버튼 추가
        if st.button("세션 재설정"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# 메인 UI
st.title("연구 에이전트")
st.write("원하는 주제에 대해 질문하세요. 에이전트가 검색하고 답변해드립니다.")

# 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 처리 중 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("생각 중...")

        # thread_id 및 assistant_id 확인
        if not st.session_state.thread_id or not st.session_state.assistant_id:
            message_placeholder.write(
                "어시스턴트 또는 쓰레드가 설정되지 않았습니다. 세션을 재설정하고 다시 시도해주세요."
            )
        else:
            try:
                # 메시지 전송
                send_message(
                    st.session_state.thread_id, prompt, st.session_state.openai_api_key
                )

                # 런 생성
                client.api_key = st.session_state.openai_api_key
                run = client.beta.threads.runs.create(
                    thread_id=st.session_state.thread_id,
                    assistant_id=st.session_state.assistant_id,
                )
                st.session_state.run_id = run.id

                # 실행 상태 확인
                while True:
                    run_status = check_run_status(
                        st.session_state.thread_id,
                        st.session_state.run_id,
                        st.session_state.openai_api_key,
                    )

                    if run_status.status == "completed":
                        # 완료된 경우 메시지 가져오기
                        messages = get_messages(
                            st.session_state.thread_id, st.session_state.openai_api_key
                        )

                        # 최신 메시지 (첫 번째 메시지)
                        latest_message = next(iter(messages))
                        if latest_message.role == "assistant":
                            assistant_response = latest_message.content[0].text.value
                            message_placeholder.write(assistant_response)

                            # 세션 상태에 메시지 추가
                            st.session_state.messages.append(
                                {"role": "assistant", "content": assistant_response}
                            )
                        break

                    elif run_status.status == "requires_action":
                        # 도구 호출 필요
                        tool_calls = (
                            run_status.required_action.submit_tool_outputs.tool_calls
                        )
                        handle_tool_calls(
                            st.session_state.thread_id,
                            st.session_state.run_id,
                            tool_calls,
                            st.session_state.openai_api_key,
                        )

                    elif run_status.status == "failed":
                        message_placeholder.write(
                            f"오류가 발생했습니다: {run_status.last_error}"
                        )
                        break

                    # 잠시 대기
                    time.sleep(1)

            except Exception as e:
                message_placeholder.write(f"오류가 발생했습니다: {str(e)}")
