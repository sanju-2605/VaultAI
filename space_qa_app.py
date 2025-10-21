import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

st.title("SpaceQA Test")

# Load model once
tokenizer_qa = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model_qa = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

def get_answer(question, context):
    inputs = tokenizer_qa(question, context, return_tensors="pt")
    outputs = model_qa(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)
    input_ids = inputs["input_ids"][0]
    answer_ids = input_ids[start_index:end_index+1]
    answer = tokenizer_qa.decode(answer_ids, skip_special_tokens=True)
    return answer

# This ensures chat history is saved across runs
if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.text_input("Your question:", key="question_input")
context = st.text_area("Your context passage:", key="context_input")

if question and context:
    answer = get_answer(question, context)
    st.session_state["history"].append((question, answer))
    st.write(f"Answer: {answer}")

st.write("Previous Q&A:")
for q, a in st.session_state["history"]:
    st.write(f"Q: {q}")
    st.write(f"A: {a}")
    st.write("---")