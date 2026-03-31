import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model (use flan-t5 for better stability)
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Functions
def summarize(text):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def correct_grammar(text):
    input_text = "fix grammar: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer_question(context, question):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI
st.set_page_config(page_title="T5 AI Assistant", layout="centered")

st.title("🧠 T5 AI Text Intelligence Assistant")

task = st.selectbox("Choose Task", ["Summarization", "Grammar Correction", "Question Answering"])

if task == "Summarization":
    text = st.text_area("Enter text to summarize")
    if st.button("Summarize"):
        if text.strip() != "":
            result = summarize(text)
            st.success(result)

elif task == "Grammar Correction":
    text = st.text_area("Enter sentence")
    if st.button("Correct Grammar"):
        if text.strip() != "":
            result = correct_grammar(text)
            st.success(result)

elif task == "Question Answering":
    context = st.text_area("Enter context")
    question = st.text_input("Enter question")
    if st.button("Get Answer"):
        if context.strip() != "" and question.strip() != "":
            result = answer_question(context, question)
            st.success(result)