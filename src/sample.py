import torchaudio

torchaudio.set_audio_backend("soundfile")

waveform, sr = torchaudio.load(
    r"C:\Users\kotag\OneDrive\Desktop\GrammarScoreGenerator\data\dataset\audios_train\audio_2.wav"
)

print(f"Sample Rate: {sr}")
print(f"Waveform Shape: {waveform.shape}")
