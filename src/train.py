import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
from tqdm import tqdm

from dataset import AudioDataset
from features import Wav2Vec2FeatureExtractor
from model import GrammarScorer
#paths
CSV_PATH= 'dataset/csvs/train.csv'
AUDIO_DIR='dataset/audios/train_wav'
#hyperparameters
BATCH_SIZE=8
EPOCHS=15
LR=1e-3
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def main():
    dataset= AudioDataset(CSV_PATH,AUDIO_DIR)
    train_size=int(0.8*len(dataset))
    val_size=len(dataset)-train_size
    train_ds,val_ds=random_split(dataset,[train_size,val_size])
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE)

    feature_extractor=Wav2Vec2FeatureExtractor()
    model=GrammarScorer().to(DEVICE)

    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=LR)

    best_val_loss=float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss=0.0
        for waveforms,labels in tqdm(train_loader,desc=f"Epoch {epoch+1}"):
            embeddings=[]
            for wf in waveforms:
                emb=feature_extractor.extract(wf)
                embeddings.append(emb)
            embeddings=torch.stack(embeddings).to(DEVICE)
            labels=labels.to(DEVICE)

            optimizer.zero_grad()
            preds=model(embeddings)
            loss=criterion(preds,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        train_loss/=len(train_loader)
        #validation
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for waveforms,labels in val_loader:
                embeddings=[]
                for wf in waveforms:
                    embeddings.append(feature_extractor.extract(wf))
                embeddings=torch.stack(embeddings).to(DEVICE)
                labels=labels.to(DEVICE)
                preds=model(embeddings)
                loss=criterion(preds,labels)
                val_loss+=loss.item()
            val_loss/=len(val_loader)
            print(
                f"Epoch{epoch+1}:"
                f"Train Loss:{train_loss:.4f},"
                f"Val Loss:{val_loss:.4f}"
            )
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                torch.save(model.state_dict(),"best_model.pt")
        print("Training complete. Best Val Loss:",best_val_loss)
if __name__=="__main__":
    main()