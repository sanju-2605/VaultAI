import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

st.title("SpaceQA: PDF, Summarization & Q&A")

# Load QA model once
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
pdf_file = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_uploader")
pdf_content = ""
if pdf_file:
    pdf_content = extract_text_from_pdf(pdf_file)

    # Only show extracted content when PDF present, disabled for input
    st.text_area("Extracted PDF text", pdf_content, height=200, key="pdf_content_area", disabled=True)

# Summarize PDF (button and display block only if PDF loaded)
if pdf_content:
    if st.button("Summarize PDF", key="summarize_btn"):
        with st.spinner("Summarizing..."):
            summary = summarizer(pdf_content[:1000])[0]['summary_text']
        st.write("Summary:", summary)

st.header("Ask a Question")

# Context selection
context_input_choice = st.radio("Use which context?", ["Manual", "Extracted PDF"], key="context_choice_radio")
if context_input_choice == "Manual":
    context = st.text_area("Context passage for QA", key="manual_context_area")
else:
    context = pdf_content

# Only ONE text_input for question, always with the same key
question = st.text_input("Your question:", key="unique_qa_question")

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
