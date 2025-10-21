import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0a2337 0%, #1e0859 100%);
    color: #fafbfc;
}
.stApp {
    background: linear-gradient(135deg, #0a2337 0%, #1e0859 100%);
}
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #80ffc8 !important;
    font-family: 'Orbitron', Arial, sans-serif;
}
.stTextInput, .stTextArea, .stFileUploader, .stButton>button {
    background: rgba(30, 8, 89, 0.3);
    color: #fafbfc;
    border-radius: 1em;
    border: 1.5px solid #3980fa;
    font-family: 'Orbitron', Arial, sans-serif;
}
.stButton>button {
    background: linear-gradient(90deg, #3980fa 0%, #310084 100%);
    color: #80ffc8;
    border-radius: 1em;
    font-family: 'Orbitron', Arial, sans-serif;
}
.stRadio>div>label {
    color: #fafbfc !important;
    font-family: 'Orbitron', Arial, sans-serif;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.title("🚀 SpaceQA: Your Gemini Chatbot")

# RBAC Login block (all keys unique to login)
if "role" not in st.session_state:
    st.session_state["role"] = None
if st.session_state["role"] is None:
    st.subheader("Login for RBAC")
    login_username = st.text_input("Username", key="login_username_input")
    login_password = st.text_input("Password", type="password", key="login_password_input")
    if st.button("Login", key="login_button"):
        if login_username == "admin" and login_password == "adminpass":
            st.session_state["role"] = "admin"
            st.success("Logged in as admin")
            st.rerun()
        elif login_username == "user" and login_password == "userpass":
            st.session_state["role"] = "user"
            st.success("Logged in as user")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

if st.session_state["role"] == "admin":
    st.info("You have full (admin) access!")
else:
    st.info("You have user access (summarize, ask questions, view history)")

@st.cache_resource
def load_qa():
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    return tokenizer, model
tokenizer_qa, model_qa = load_qa()

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer = load_summarizer()

def get_answer(question, context):
    inputs = tokenizer_qa(question, context, return_tensors="pt")
    outputs = model_qa(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)
    input_ids = inputs["input_ids"][0]
    answer_ids = input_ids[start_index:end_index+1]
    answer = tokenizer_qa.decode(answer_ids, skip_special_tokens=True)
    return answer

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

if "history" not in st.session_state:
    st.session_state["history"] = []

st.header("Upload a PDF")
pdf_file = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_file_uploader")
pdf_content = ""
if pdf_file:
    pdf_content = extract_text_from_pdf(pdf_file)
    st.text_area("Extracted PDF text", pdf_content, height=200, key="pdf_extracted_text", disabled=True)

if pdf_content:
    if st.button("Summarize PDF", key="pdf_summarize_button"):
        with st.spinner("Summarizing..."):
            summary = summarizer(pdf_content[:1000])[0]['summary_text']
        st.write("Summary:", summary)

st.header("Ask a Question")
context_choice = st.radio("Use which context?", ["Manual", "Extracted PDF"], key="question_context_choice")
if context_choice == "Manual":
    context = st.text_area("Context passage for QA", key="qa_manual_context_input")
else:
    context = pdf_content

question = st.text_input("Your question:", key="qa_question_input")
if question and context:
    with st.spinner("Answering..."):
        answer = get_answer(question, context)
    st.session_state["history"].append((question, answer))
    st.write(f"Answer: {answer}")

if st.session_state["history"]:
    st.write("Previous Q&A:")
    for q, a in st.session_state["history"]:
        st.write(f"Q: {q}")
        st.write(f"A: {a}")
        st.write("---")

if st.session_state["role"] == "admin":
    st.subheader("Admin Only Tools")
    if st.button("Clear Q&A History", key="admin_clear_history_btn"):
        st.session_state["history"].clear()
        st.success("History cleared!")