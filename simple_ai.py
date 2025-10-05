# HistoBot + Streamlit - Fixed Version
import subprocess
import sys

# -----------------------------
# 1. Auto-install packages
# -----------------------------
required_packages = ['numpy', 'sentence-transformers', 'requests', 'beautifulsoup4', 'scikit-learn', 'streamlit']
for package in required_packages:
    try:
        __import__(package.split('-')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# -----------------------------
# 2. Imports
# -----------------------------
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -----------------------------
# 3. Load notes
# -----------------------------
def load_notes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

docs = load_notes(r"C:\Users\TheoKevinMasie\Desktop\history_notes.txt")
sentences = [s for s in docs.split('\n') if s.strip() != '']

if not sentences:
    raise ValueError("No sentences found in notes.")

print("Notes loaded successfully!")

# -----------------------------
# 4. Load embedding model
# -----------------------------
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# -----------------------------
# 5. Create embeddings safely
# -----------------------------
embeddings = model.encode(sentences, convert_to_numpy=True)
embeddings = np.array(embeddings)

# Ensure 2D for single sentence
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(1, -1)

# Normalize
doc_vecs = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# -----------------------------
# 6. Retrieve function
# -----------------------------
def retrieve(query, top_k=1):
    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = np.array(query_vec)
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    
    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    results = [sentences[i] for i in top_indices]
    return " ".join(results) if results else ""

# -----------------------------
# 7. Optional online search
# -----------------------------
def online_search(query):
    try:
        url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs[:3]])
        return text.strip() if text.strip() else "No online info found."
    except Exception:
        return "No online info found."

# -----------------------------
# 8. Ask HistoBo
