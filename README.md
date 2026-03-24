# 🎥 RAG-Based AI Video Assistant

This project is a **Retrieval-Augmented Generation (RAG)** system that allows you to:

* 🔍 Ask questions about a video
* 🧠 Get AI-generated answers
* ⏱ See exact timestamps where topics are discussed
* 🎥 Jump directly to that part of the video

---

## 🚀 Features

* Semantic search using **FAISS**
* Embeddings using **bge-m3 (Ollama)**
* Local LLM (**phi3**) for answering questions
* Timestamp-based video navigation
* Handles:

  * Q&A
  * Summaries
  * "Where is this taught?" queries

---

## 🧠 Tech Stack

* Python
* FAISS
* Ollama (phi3 + bge-m3)
* NumPy
* Requests

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

https://ollama.com

### 3. Pull models

```bash
ollama pull phi3
ollama pull bge-m3
```

---

## ▶️ Run

```bash
python rag_chatbot.py
```

---

## 💬 Example Queries

* What is a process?
* Give summary of the course
* Where is operating system taught?

---

## 📌 Notes

* Large files (audio/video/embeddings) are excluded from repo
* You need to generate embeddings locally

---

## 🔥 Future Improvements

* Streamlit UI with video player
* Clickable timestamps
* Better chunking
* Smart query detection

---

## 👨‍💻 Author

Aakarshika Rai
