from dataset import AudioDataset
DATASET_CSV="dataset/csvs/train.csv"
AUDIO_DIR="dataset/audios/train_wav"
dataset=AudioDataset(DATASET_CSV,AUDIO_DIR)
print(f"Dataset size: {len(dataset)}")
waveform,label=dataset[0]
print(f"Waveform shape: {waveform.shape}, Label: {label}")  
