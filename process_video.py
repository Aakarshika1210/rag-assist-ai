#converts the video to mp3
import os
import subprocess

# make sure output folder exists
os.makedirs("audios", exist_ok=True)

input_file = "videos/videoplayback.mp4"
output_file = "audios/videoplayback.mp3"

subprocess.run([
    "ffmpeg",
    "-i", input_file,
    "-vn",        # remove video
    "-q:a", "0",  # best quality mp3
    output_file
])

print("✅ MP3 extracted successfully!")