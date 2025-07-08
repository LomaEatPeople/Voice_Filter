import os
import numpy as np
import librosa
import tensorflow as tf

# โหลดโมเดลที่ฝึกไว้
model = tf.keras.models.load_model("dog_sound_model.h5")

# ฟังก์ชันแปลงเสียงเป็น log-mel spectrogram
def extract_log_mel(path, max_len=128):
    y, sr = librosa.load(path, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel)
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_len]
    return log_mel[..., np.newaxis]

# ฟังก์ชันทำนายเสียง
def predict_dog_sound(filepath):
    mel = extract_log_mel(filepath)
    mel = np.expand_dims(mel, axis=0)  # เพิ่ม batch dimension
    prediction = model.predict(mel)[0][0]
    return prediction

# ----------------------------
# ใช้งาน (ใส่ path ของไฟล์เสียงที่ต้องการทำนาย)
file_to_predict = "Dog_DataSet/1-30226-A-0.wav"  # ← เปลี่ยนตรงนี้เป็นไฟล์ของคุณ
score = predict_dog_sound(file_to_predict)
print(f"✅ File: {file_to_predict}")
print(f"🎯 Confidence (เสียงหมา): {score:.2f}")
print("🐶 เป็นเสียงหมา" if score >= 0.5 else "❌ ไม่ใช่เสียงหมา")
