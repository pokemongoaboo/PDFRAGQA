import streamlit as st
import PyPDF2
from openai import OpenAI
from io import BytesIO

# 設置OpenAI客戶端
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_openai_response(prompt, context):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一個有幫助的助手。請使用以下上下文來回答用戶的問題。"},
                {"role": "user", "content": f"上下文: {context}\n\n任務: {prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"發生錯誤: {str(e)}")
        return None

def summarize_text(text):
    prompt = f"請用約100字簡要概括以下文本的內容：\n\n{text[:4000]}"
    return get_openai_response(prompt, "")

def generate_questions(text):
    prompt = "根據給定的文本，生成3到5個相關的問題。請以編號列表的形式提供這些問題。"
    return get_openai_response(prompt, text[:4000])

@st.cache_data
def process_pdf(pdf_file):
    text = extract_text_from_pdf(BytesIO(pdf_file.read()))
    summary = summarize_text(text)
    questions = generate_questions(text)
    return text, summary, questions

st.title("PDF 智能問答系統")

uploaded_file = st.file_uploader("請選擇一個PDF檔案", type="pdf")

if uploaded_file is not None:
    pdf_content, summary, suggested_questions = process_pdf(uploaded_file)
    st.success("PDF上傳並處理成功！")

    st.subheader("文件摘要")
    st.write(summary)

    st.subheader("您可能想問的問題：")
    questions = suggested_questions.split("\n")
    selected_question = st.radio("選擇一個問題或在下方輸入您自己的問題：", 
                                 questions + ["我想問自己的問題"])

    if selected_question == "我想問自己的問題":
        user_question = st.text_input("在此輸入您的問題：")
    else:
        user_question = selected_question

    if user_question and user_question != "我想問自己的問題":
        with st.spinner("正在生成答案..."):
            answer = get_openai_response(user_question, pdf_content)
        if answer:
            st.subheader("答案：")
            st.write(answer)

st.sidebar.write("本應用程式使用OpenAI的API進行智能問答。")
st.sidebar.info("注意：本應用程式需要OpenAI API金鑰才能運作。")
