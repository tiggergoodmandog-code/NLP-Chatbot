import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# โหลดโมเดล
model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline('text-generation', model='distilgpt2')

# Corpus ข้อมูล Python
corpus = [
    "Python uses dynamic typing, meaning you don't need to declare variable types explicitly.",
    "A list in Python can store multiple data types within the same structure.",
    "A tuple is similar to a list but is immutable and cannot be modified after creation.",
    "A dictionary stores data in key-value pairs and provides fast lookups.",
    "A set is an unordered collection that does not allow duplicate items.",
    "Functions in Python are defined using the 'def' keyword and can return values.",
    "A lambda function is an anonymous, inline function used for short operations.",
    "List comprehension provides a concise way to create lists and is faster than a regular loop.",
    "Python uses indentation to define code blocks instead of braces.",
    "A module is a .py file containing functions, classes, or variables that can be imported.",
    "A package is a directory containing multiple modules and an __init__.py file.",
    "Try-except blocks are used to handle exceptions and prevent program crashes.",
    "The 'with' statement is used to manage resources, such as automatically opening and closing files.",
    "A generator uses the 'yield' keyword to create an iterator with low memory usage.",
    "A decorator modifies or extends the behavior of a function without changing its code.",
    "A class in Python is defined using the 'class' keyword and is used to create objects.",
    "The 'self' parameter refers to the current instance of a class.",
    "__init__ is the constructor method that runs automatically when an object is created.",
    "A virtual environment is used to isolate project dependencies.",
    "pip is the standard package manager for installing and managing Python packages."
]

# Encode corpus
corpus_embeddings = model.encode(corpus)

# ฟังก์ชันหาความเหมือน
def cosine_similarity(query_embedding, corpus_embeddings):
    dot_product = np.dot(corpus_embeddings, query_embedding.T)
    query_norm = np.linalg.norm(query_embedding)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    return dot_product / (corpus_norms * query_norm)

# ฟังก์ชันดึงข้อมูลที่เกี่ยวข้องจาก corpus
def retrieve(query, top_k=2):
    query_embedding = model.encode(query)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return [corpus[i] for i in top_k_indices]

# ฟังก์ชันสร้างคำตอบ
def generate_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question based on the context below:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text'].strip()

# Streamlit UI
st.set_page_config(page_title="Python Chatbot", page_icon=":robot:", layout="wide")
st.title(":robot: Python Programming Chatbot")
st.write("Ask me anything about Python programming!")

# Input
user_input = st.text_input("Your question:", "")

if st.button("Ask") and user_input:
    with st.spinner("Generating answer..."):
        answer = generate_answer(user_input)
    st.markdown(f"**Chatbot:** {answer}")
