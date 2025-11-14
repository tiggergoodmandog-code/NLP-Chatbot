import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# LLM generator
generator = pipeline('text-generation', model='distilgpt2')

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

# Retriever
def retrieve(query, top_k=2):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity(query_embedding, corpus_embeddings)

    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return [corpus[i] for i in top_k_indices]

# Answer generation using context
def generate_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question based only on the context below:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = generator(prompt, max_length=120, num_return_sequences=1)
    return response[0]['generated_text'].strip()

# Chat loop
def main():
    print("Hello! I'm your Python chatbot. Ask me anything about Python programming.")

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        response = generate_answer(query)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
