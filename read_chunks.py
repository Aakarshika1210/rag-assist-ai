import requests
import json
import pandas as pd

# -------------------------------
# STEP 1: Load chunks
# -------------------------------
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -------------------------------
# STEP 2: Create embeddings
# -------------------------------
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    return r.json()["embeddings"]

# extract all text
texts = [chunk["text"] for chunk in chunks]

print("🚀 Creating embeddings...")

embeddings = create_embedding(texts)

# -------------------------------
# STEP 3: Attach embeddings
# -------------------------------
for i, chunk in enumerate(chunks):
    chunk["id"] = i
    chunk["embedding"] = embeddings[i]

# -------------------------------
# STEP 4: Save output
# -------------------------------
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print("✅ embeddings.json created!")

# -------------------------------
# STEP 5: Optional DataFrame
# -------------------------------
df = pd.DataFrame(chunks)
print(df.head())

import joblib

joblib.dump(df, "embeddings.joblib")

print("✅ DataFrame saved as embeddings.joblib")