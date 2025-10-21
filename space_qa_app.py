import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

st.title("SpaceQA Test")

# Load model ONCE
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

# The key argument ensures no DuplicateElementId errors
question = st.text_input("Your question:", key="question_input")
context = st.text_area("Your context passage:", key="context_input")

if question and context:
    answer = get_answer(question, context)
    st.write(f"Answer: {answer}")

