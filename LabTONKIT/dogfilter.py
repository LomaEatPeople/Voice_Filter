import pandas as pd
import os
import shutil

df = pd.read_csv("ESC-50-master/ESC-50-master/meta/esc50.csv")
dog_sounds = df[df["category"] == "dog"]

print(dog_sounds[["filename"]])

os.makedirs("Dog_DataSet", exist_ok=True)

for fname in dog_sounds["filename"]: 
    src = os.path.join("ESC-50-master/ESC-50-master/audio", fname)
    dst = os.path.join("Dog_DataSet", fname)
    shutil.copy(src, dst)


notdog_sounds = df[df["category"] != "dog"]

print(notdog_sounds[["filename"]])

os.makedirs("NotDog_DataSet", exist_ok=True)

for fname in notdog_sounds["filename"]: 
    src = os.path.join("ESC-50-master/ESC-50-master/audio", fname)
    dst = os.path.join("NotDog_DataSet", fname)
    shutil.copy(src, dst)
