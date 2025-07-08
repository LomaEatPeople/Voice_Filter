from transformers import ClapProcessor, ClapModel
import torchaudio
import torch
import librosa
import numpy as np

# โหลด CLAP model
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")

def get_audio_embedding(audio_path):
    audio_input, sr = torchaudio.load(audio_path)
    audio_input = audio_input.mean(dim=0).unsqueeze(0)  # mono
    inputs = processor(audios=audio_input, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_audio_features(**inputs)
    return embeddings[0]

# โหลดเสียงหมาเห่า (reference)
dog_embed = get_audio_embedding("dog_bark_sample.wav")

# โหลดเสียงผสม
mix_embed = get_audio_embedding("mixed_audio.wav")

# วัดความคล้าย
similarity = torch.nn.functional.cosine_similarity(dog_embed, mix_embed, dim=0)
print("คล้ายกันแค่ไหน:", similarity.item())
