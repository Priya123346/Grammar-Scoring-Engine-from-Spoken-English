import torch
from torch.utils.data import DataLoader
import pandas as pd

from dataset import AudioDataset
from features import Wav2Vec2FeatureExtractor   
from model import GrammarScorer

CSV_PATH='dataset/csvs/test.csv'
AUDIO_DIR='dataset/audios/test_wav'
MODEL_PATH='best_model.pt'
DEVICE='cuda' if torch.cuda.is_available() else "cpu"
OUTPUT_CSV='predictions.csv'

dataset=AudioDataset(CSV_PATH,AUDIO_DIR,has_labels=False)
loader=DataLoader(dataset, batch_size=8)    

feature_extractor=Wav2Vec2FeatureExtractor()
model=GrammarScorer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
model.eval()

preds=[]
with torch.no_grad():
    for waveforms in loader:
        embeddings=[]
        for wf in waveforms:
            embeddings.append(feature_extractor.extract(wf))
        embeddings=torch.stack(embeddings).to(DEVICE)
        outputs=model(embeddings)
        preds.extend(outputs.cpu().numpy())
pd.DataFrame({'preds':preds}).to_csv(OUTPUT_CSV,index=False)