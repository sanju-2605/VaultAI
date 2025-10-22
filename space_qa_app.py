import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# --- User Storage ---
if "USERS" not in st.session_state:
    st.session_state["USERS"] = {
        "admin": {"password": "adminpass", "role": "admin"}
    }

if "role" not in st.session_state:
    st.session_state["role"] = None

# --- CSS ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #E6E6FA 0%, #D8BFD8 100%);
    color: #fafbfc;
}
.stApp {
    background: linear-gradient(135deg, #E6E6FA 0%, #D8BFD8 100%);
}
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #6e36b6 !important;
    font-family: 'Orbitron', Arial, sans-serif;
}
.stTextInput, .stTextArea, .stFileUploader, .stButton>button {
    background: #fff;
    border-radius: 1em;
    border: 1.5px solid #e6e6fa;
    font-family: 'Orbitron', Arial, sans-serif;
}
.stTextInput input, .stTextArea textarea {
    background: #fff !important;
    color: #000 !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #6e6e6e !important;
    opacity: 1 !important;
}
label, .streamlit-expanderHeader, .stRadio label, .stSelectbox label, .stTextInput label {
    color: #6e36b6 !important;
}
.stButton>button {
    background: linear-gradient(90deg, #c8a2c8 0%, #b39ddb 100%);
    color: #6e36b6;
    border-radius: 1em;
    font-family: 'Orbitron', Arial, sans-serif;
}
.stRadio>div>label {
    color: #6e36b6 !important;
    font-family: 'Orbitron', Arial, sans-serif;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.title("VaultAI: Your Secure Chatbot")

# --- Auth Interface ---
page = st.sidebar.radio("Choose Action", ["Login", "Register"], key="login_or_register_radio")

if st.session_state["role"] is None:
    if page == "Register":
        st.header("Register New User")
        new_user = st.text_input("New Username", key="reg_username")
        new_pwd = st.text_input("Choose Password", type="password", key="reg_password")
        new_role = st.selectbox("Register as", ["employee", "admin"], key="reg_role")
        if st.button("Register", key="reg_submit"):
            if new_user in st.session_state["USERS"]:
                st.error("Username already exists!")
            elif not new_user or not new_pwd:
                st.error("Please fill all fields.")
            else:
                st.session_state["USERS"][new_user] = {
                    "password": new_pwd,
                    "role": new_role
                }
                st.success(f"Registered {new_user} as {new_role}. Please switch to Login to proceed.")
        st.stop()
    elif page == "Login":
        st.header("🔐 Login to VaultAI")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_submit"):
            users = st.session_state["USERS"]
            if username in users and users[username]["password"] == password:
                st.session_state["role"] = users[username]["role"]
                st.success(f"Logged in as {username} ({users[username]['role']})")
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.stop()

# --- Role-specific messages ---
if st.session_state["role"] == "admin":
    st.info("🧠 You are logged in as an Admin. You have full access.")
elif st.session_state["role"] == "employee":
    st.info("👨‍💻 You are logged in as an Employee. You can upload PDFs, summarize, and ask questions.")

# --- Model loaders ---
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

# --- Q&A ---
def get_answer(question, context):
    max_context_length = 500
    context = context[:max_context_length]
    max_question_length = 100
    question = question[:max_question_length]
    inputs = tokenizer_qa(question, context, return_tensors="pt", truncation=True, max_length=512)
    outputs = model_qa(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)
    input_ids = inputs["input_ids"][0]
    answer_ids = input_ids[start_index:end_index+1]
    answer = tokenizer_qa.decode(answer_ids, skip_special_tokens=True)
    return answer

# --- PDF extract ---
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

# --- Q&A session history ---
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
