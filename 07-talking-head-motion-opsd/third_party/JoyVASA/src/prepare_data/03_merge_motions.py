import json
import pickle
import os

label_name = "./train.json"
with open(label_name, 'r', encoding='utf-8') as file:
    train_data = json.load(file)
label_name = "./test.json"
with open(label_name, 'r', encoding='utf-8') as file:
    val_data = json.load(file)
data = train_data + val_data

save_name = "motions.pkl"
all_items = {}
for item in data:
    key = item["audio_name"]
    motions = pickle.load(open(item["motion_name"], 'rb'))
    value = motions

    all_items[key] = value
pickle.dump(all_items, open(save_name, 'wb'))