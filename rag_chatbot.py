import joblib
import numpy as np
import faiss
import requests
import webbrowser
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")
API_KEY = os.getenv("OPENROUTER_API_KEY")


VIDEO_URL = "https://www.youtube.com/watch?v=86kN-pdxsl0"

# -------------------------------
# STEP 1: Load data
# -------------------------------
df = joblib.load("embeddings.joblib")

embeddings = np.vstack(df["embedding"].values).astype("float32")
faiss.normalize_L2(embeddings)

# -------------------------------
# STEP 2: Create FAISS index
# -------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("✅ FAISS ready!")

# -------------------------------
# STEP 3: Embedding function
# -------------------------------
def create_embedding(text):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": [text]
        }
    )
    return np.array(r.json()["embeddings"]).astype("float32")

# -------------------------------
# Helper: format timestamp
# -------------------------------
def format_time(seconds):
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"

def open_video_at(seconds):
    url = f"{VIDEO_URL}&t={int(seconds)}s"
    webbrowser.open(url)




def generate_answer(prompt):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG Chatbot"
        },
        json={
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    data = response.json()

    # 🔥 DEBUG (temporary)
    if "choices" not in data:
        print("\n❌ API ERROR RESPONSE:\n", data)
        return "Error: Failed to get response from API."

    return data["choices"][0]["message"]["content"]
# STEP 4: Chat loop
# -------------------------------
while True:
    query = input("\n💬 Ask: ").strip().lower()

    if query == "exit":
        break

    # -------------------------------
    # FIX COMMON TYPOS
    # -------------------------------
    query = query.replace("syatem", "system")
    query = query.replace("couse", "course")
    query = query.replace("waht", "what")
    query = query.replace("teh", "the")

    query_lower = query

    # -------------------------------
    # STEP 5: Detect query types
    # -------------------------------
    is_timestamp = (
        "where" in query_lower
        or "when" in query_lower
        or "time" in query_lower
        or "taught" in query_lower
    )

    is_summary = (not is_timestamp) and any(word in query_lower for word in [
        "summary", "topics", "overview", "course"
    ])

    # -------------------------------
    # STEP 6: Create embedding
    # -------------------------------
    query_embedding = create_embedding(query)
    faiss.normalize_L2(query_embedding)

    # -------------------------------
    # STEP 7: Search
    # -------------------------------
    top_k = 20 if is_summary else 8
    distances, indices = index.search(query_embedding, top_k)

    retrieved = df.iloc[indices[0]].to_dict("records")

    # -------------------------------
    # DEBUG
    # -------------------------------
    print("\n🔍 Retrieved Context:\n")
    for r in retrieved[:5]:
        print("-", r["text"][:80], "...")

    # -------------------------------
    # HANDLE LOCATION QUESTIONS
    # -------------------------------
    if is_timestamp:
        print("\n⏱ Topic found at:\n")

        for r in retrieved[:5]:   # 🔥 NO FILTERING
            start = format_time(r["start"])
            end = format_time(r["end"])
            print(f"{start} - {end} → {r['text'][:80]}...")

        continue

    # -------------------------------
    # STEP 9: Build context
    # -------------------------------
    if is_summary:
        context = "\n".join([r["text"] for r in retrieved[:15]])
    else:
        context = "\n".join([r["text"] for r in retrieved[:5]])

    # -------------------------------
    # STEP 10: Prompt
    # -------------------------------
    if is_summary:
        prompt = f"""
You are an expert teacher.

IMPORTANT:
- Always answer the question.
- Do NOT say "the context does not define".
- Use your knowledge if needed.

Context:
{context}

Question:
{query}

Give a structured summary:
"""
    else:
        prompt = f"""
You are an expert computer science teacher.

IMPORTANT:
- Always answer the question.
- Do NOT say "the context does not define".
- Use your knowledge if needed.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = generate_answer(prompt)

    # -------------------------------
    # STEP 12: Print answer
    # -------------------------------
    print("\n🤖 Answer:\n")
    print(answer)

    # -------------------------------
    # STEP 13: Show timestamps
    # -------------------------------
    print("\n⏱ Relevant timestamps:\n")

    for r in retrieved[:3]:   # 🔥 NO FILTERING
        start = format_time(r["start"])
        end = format_time(r["end"])
        print(f"{start} - {end} → {r['text'][:80]}...")
        
        choice = input("Open this timestamp? (y/n): ")
        if choice.lower() == "y":

          open_video_at(r["start"])




      