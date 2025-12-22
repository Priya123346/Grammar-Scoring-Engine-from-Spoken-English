import os
import subprocess
from pathlib import Path

INPUT_DIR="data/unprocessed_data"
OUTPUT_DIR="data/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".m4a"):
        input_path=os.path.join(INPUT_DIR,file)
        output_path=os.path.join(OUTPUT_DIR,Path(file).stem+".wav")

        subprocess.run(["ffmpeg","-y","-i",input_path,"-ac","1","-ar","16000",output_path])
        print(f"Converted {input_path} to {output_path}")