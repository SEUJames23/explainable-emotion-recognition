import pandas as pd
import os
import pandas as pd

df = pd.read_csv("/mmu-audio-ssd/frontend/youjun/pretrained_models/EN_labels_gigaspeech.csv")
key2emo = dict(zip(df['Key'], df['Emotion']))

data = pd.read_csv("/mmu-audio-ssd/frontend/youjun/speechllm/results/prediction_cot.csv")


keys, pred_emos, true_emos = [], [], []
for i in range(len(data)):
    id = data.iloc[i]['id']
    key = "gigaspeech_"+os.path.basename(id)[:-4]
    cot = data.iloc[i]['cot']
    pred_emo = cot.split("with a")[2].split("tone")[0]
    pred_emos.append(pred_emo)
    keys.append(key)
    true_emo = key2emo[key]
    if true_emo == 'natural':
        true_emo = 'neutral'
    true_emos.append(true_emo)

cnt = 0
for i in range(len(pred_emos)):
    print("pred:", pred_emos[i], "true:", true_emos[i])
    if pred_emos[i].strip(' ') == true_emos[i].strip(' '):
        cnt += 1

print("Acc:", cnt/len(pred_emos))
