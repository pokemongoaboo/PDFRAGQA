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
                {"role": "user", "content": f"Context: {context}\n\nTask: {prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def summarize_text(text):
    prompt = f"Please provide a brief summary of the following text in about 100 words:\n\n{text[:4000]}"
    return get_openai_response(prompt, "")

def generate_questions(text):
    prompt = "Based on the given text, generate 3 to 5 relevant questions that could be asked about its content. Provide these questions in a numbered list."
    return get_openai_response(prompt, text[:4000])

@st.cache_data
def process_pdf(pdf_file):
    text = extract_text_from_pdf(BytesIO(pdf_file.read()))
    summary = summarize_text(text)
    questions = generate_questions(text)
    return text, summary, questions

st.title("Enhanced PDF RAG Application")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_content, summary, suggested_questions = process_pdf(uploaded_file)
    st.success("PDF uploaded and processed successfully!")

    st.subheader("Summary")
    st.write(summary)

    st.subheader("You might want to ask:")
    questions = suggested_questions.split("\n")
    selected_question = st.radio("Select a question or type your own below:", 
                                 questions + ["I want to ask my own question"])

    if selected_question == "I want to ask my own question":
        user_question = st.text_input("Type your question here:")
    else:
        user_question = selected_question

    if user_question and user_question != "I want to ask my own question":
        with st.spinner("Generating answer..."):
            answer = get_openai_response(user_question, pdf_content)
        if answer:
            st.subheader("Answer:")
            st.write(answer)

st.sidebar.write("This app uses OpenAI's API for RAG functionality.")
st.sidebar.info("Note: This app requires an OpenAI API key to function.")
