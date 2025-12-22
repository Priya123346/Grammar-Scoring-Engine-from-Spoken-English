from dataset import AudioDataset
from features import Wav2Vec2FeatureExtractor

DATASET_CSV='data/dataset/train.csv'
AUDIO_DIR='data/dataset/audios_train'

dataset=AudioDataset(DATASET_CSV,AUDIO_DIR)
feature_extractor=Wav2Vec2FeatureExtractor()
waveform,label=dataset[0]
embeddings=feature_extractor.extract(waveform)
print("embeddings shape:",embeddings.shape)
print("label:",label)