import torch 
from transformers import Wav2Vec2Model,Wav2Vec2Processor
class Wav2Vec2FeatureExtractor:
    def __init__(self,model_name="facebook/wav2vec2-base"):
        self.processor=Wav2Vec2Processor.from_pretrained(model_name)
        self.model=Wav2Vec2Model.from_pretrained(model_name,use_safetensors=True)
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.eval()
    def extract(self,waveform:torch.Tensor)->torch.Tensor:
        with torch.no_grad():
            inputs=self.processor(waveform,sampling_rate=16000,return_tensors="pt")
            outputs=self.model(**inputs)
            hidden_states=outputs.last_hidden_state
            embeddings=hidden_states.mean(dim=1).squeeze(0)
        return embeddings