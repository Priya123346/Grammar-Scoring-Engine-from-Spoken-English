from dataset import AudioDataset
DATASET_CSV="data/dataset/train.csv"
AUDIO_DIR="data/dataset/audios_train"
dataset=AudioDataset(DATASET_CSV,AUDIO_DIR)
print(f"Dataset size: {len(dataset)}")
waveform,label=dataset[0]
print(f"Waveform shape: {waveform.shape}, Label: {label}")  
