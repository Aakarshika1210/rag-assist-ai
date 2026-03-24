import whisper
import json

# -------------------------------
# STEP 1: Load model
# -------------------------------
model = whisper.load_model("base")

# -------------------------------
# STEP 2: Transcribe
# -------------------------------
result = model.transcribe(
    audio="audios/videoplayback.mp3",
    language="hi",
    task="translate",
    verbose=True
)

# -------------------------------
# STEP 3: Create 10-sec chunks
# -------------------------------
chunks = []
current_chunk = {
    "start": 0,
    "end": 10,
    "text": ""
}

for segment in result["segments"]:
    seg_start = segment["start"]
    seg_end = segment["end"]
    seg_text = segment["text"].strip()

    while seg_start < seg_end:
        # If segment fits inside current chunk
        if seg_start < current_chunk["end"]:
            current_chunk["text"] += " " + seg_text
            break
        else:
            # Save current chunk
            chunks.append(current_chunk)

            # Move to next 10-sec window
            current_chunk = {
                "start": current_chunk["end"],
                "end": current_chunk["end"] + 10,
                "text": ""
            }

# Add last chunk
if current_chunk["text"]:
    chunks.append(current_chunk)

# -------------------------------
# STEP 4: Save JSON
# -------------------------------
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)

print("✅ 10-sec chunks saved to chunks.json")