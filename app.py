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
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

@st.cache_data
def process_pdf(pdf_file):
    return extract_text_from_pdf(BytesIO(pdf_file.read()))

st.title("PDF RAG Application")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_content = process_pdf(uploaded_file)
    st.success("PDF uploaded and processed successfully!")

    user_question = st.text_input("Ask a question about the PDF content:")
    
    if user_question:
        with st.spinner("Generating answer..."):
            answer = get_openai_response(user_question, pdf_content)
        if answer:
            st.write("Answer:", answer)

st.sidebar.write("This app uses OpenAI's API for RAG functionality.")
st.sidebar.info("Note: This app requires an OpenAI API key to function.")
