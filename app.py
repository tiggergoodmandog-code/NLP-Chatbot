import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load LLM generator
@st.cache_resource
def load_generator_model():
    return pipeline('text-generation', model='distilgpt2')

model = load_embedding_model()
generator = load_generator_model()

# Knowledge base corpus
corpus = [
    "Python is one of the most popular programming languages for beginners and professionals.",
    "I started learning Python because it's easy to read and has a clean syntax.",
    "Functions in Python help organize code and make it reusable.",
    "Python supports multiple programming paradigms, including object-oriented and functional programming.",
    "Many data scientists prefer Python because of libraries like NumPy and pandas.",
    "Indentation is very important in Python since it defines code blocks.",
    "I used Python to write a small script that automates file management on my computer.",
    "Python dictionaries store data as key-value pairs, making lookups very efficient.",
    "The Python community is huge, which makes finding tutorials and resources very easy.",
    "I’m practicing how to use Python loops to iterate through lists and dictionaries.",
    "Exception handling in Python helps prevent programs from crashing unexpectedly.",
    "Python’s list comprehension feature allows concise and readable list creation.",
    "I learned how to install Python packages using pip.",
    "Object-oriented programming in Python involves using classes and objects.",
    "I wrote a Python program that fetches data from an API and displays it nicely.",
    "Virtual environments in Python help manage different project dependencies.",
    "Python is widely used in machine learning through frameworks like TensorFlow and PyTorch.",
    "I enjoy using Python because it lets me build projects quickly.",
    "The Python print() function is one of the first things beginners learn.",
    "I want to improve my Python skills so I can build more advanced applications."
]

corpus_embeddings = model.encode(corpus)

# Cosine similarity
def cosine_similarity(query_embeddings, corpus_embeddings):
    dot_product = np.dot(corpus_embeddings, query_embeddings.T)
    query_norm = np.linalg.norm(query_embeddings)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    return dot_product / (corpus_norms * query_norm)

# Retrieve similar documents
def retrieve(query, top_k=2):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity(query_embedding, corpus_embeddings)
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return [corpus[i] for i in top_k_indices]

# Generate final answer
def generate_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question based only on the context below:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=120, num_return_sequences=1)
    return response[0]['generated_text'].strip()

# -------------------------- Streamlit UI --------------------------

st.title("🐍 Python Programming RAG Chatbot")
st.write("Ask anything about Python programming!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Input box
query = st.chat_input("Type your question here...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    # Generate bot response
    answer = generate_answer(query)

    # Save bot message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
