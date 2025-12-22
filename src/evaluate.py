import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import AudioDataset
from features import Wav2Vec2FeatureExtractor
from model import GrammarScorer

CSV_PATH='data/dataset/train.csv'
AUDIO_DIR='data/dataset/audios_train'
MODEL_PATH='best_model.pt'
DEVICE='cuda' if torch.cuda.is_available() else "cpu"

def main():
    dataset=AudioDataset(CSV_PATH,AUDIO_DIR)
    loader=DataLoader(dataset, batch_size=8)
    feature_extractor=Wav2Vec2FeatureExtractor()
    model=GrammarScorer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    model.eval()
    preds=[]
    targets=[]
    with torch.no_grad():
        for waveforms,labels in loader:
            embeddings=[]
            for wf in waveforms:
                embeddings.append(feature_extractor.extract(wf))
            embeddings=torch.stack(embeddings).to(DEVICE)
            outputs=model(embeddings)
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.numpy())
    preds=np.array(preds)
    targets=np.array(targets)
    mse=np.mean((preds-targets)**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(preds-targets))

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
if __name__=="__main__":
    main()