import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

def extract_log_mel(path, max_len=128):
    y, sr = librosa.load(path, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel)
    
    # pad/truncate เพื่อให้ spectrogram มีขนาดเท่ากัน
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_len]
    return log_mel

# เตรียมข้อมูล
X = []
y = []

# เสียงหมา
for file in os.listdir("Dog_DataSet"):
    if file.endswith(".wav"):
        path = os.path.join("Dog_DataSet", file)
        mel = extract_log_mel(path)
        X.append(mel)
        y.append(1)

# เสียงอื่นๆ
for file in os.listdir("NotDog_DataSet"):
    if file.endswith(".wav"):
        path = os.path.join("NotDog_DataSet", file)
        mel = extract_log_mel(path)
        X.append(mel)
        y.append(0)

X = np.array(X)
y = np.array(y)

# Reshape เป็นรูปภาพ 128x128 (ช่องเดียว)
X = X[..., np.newaxis]

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้าง CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # แยกหมา/ไม่หมา
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ฝึกโมเดล
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# ✅ เซฟโมเดลไว้ใช้ต่อในอนาคต
model.save("dog_sound_model.h5")

# ทดสอบ
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.2f}")
