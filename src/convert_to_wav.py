import os
import subprocess
from pathlib import Path

INPUT_DIR = "dataset/audios/test"
OUTPUT_DIR = "dataset/audios/test_wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    input_path = os.path.join(INPUT_DIR, file)

    if not os.path.isfile(input_path):
        continue

    output_name = Path(file).stem + ".wav"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_path
    ])
    print(f"Converted {input_path} to {output_path}")