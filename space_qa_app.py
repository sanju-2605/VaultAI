import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

st.title("SpaceQA Test with PDF & Summarization")

# Load QA model once
tokenizer_qa = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model_qa = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

# Load summarization pipeline once
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

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
            text += page.extract_text()
        return text

if "history" not in st.session_state:
    st.session_state["history"] = []

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF to convert its content", type=["pdf"])
pdf_content = ""
if pdf_file:
    pdf_content = extract_text_from_pdf(pdf_file)
    st.text_area("Extracted PDF content:", pdf_content, height=200)

# Summarize PDF
if pdf_content:
    if st.button("Summarize PDF"):
        with st.spinner("Generating summary..."):
            summary = summarizer(pdf_content[:1000])[0]['summary_text']  # Limit length for demo
        st.write("Summary:", summary)

question = st.text_input("Your question:", key="question_input")
context_input_choice = st.radio("Use which context?", ["Manual", "Extracted PDF"])
if context_input_choice == "Manual":
    context = st.text_area("Your context passage:", key="context_input")
else:
    context = pdf_content

if question and context:
    answer = get_answer(question, context)
    st.session_state["history"].append((question, answer))
    st.write(f"Answer: {answer}")

st.write("Previous Q&A:")
for q, a in st.session_state["history"]:
    st.write(f"Q: {q}")
    st.write(f"A: {a}")
    st.write("---")