import streamlit as st

st.title("SpaceQA Test")

question = st.text_input("Your question:")
context = st.text_area("Your context passage:")

if question and context:
    st.write(f"Q: {question}")
    st.write(f"Context: {context}")

