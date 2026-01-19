import pandas as pd
import os
import pandas as pd

df = pd.read_csv("/mmu-audio-ssd/frontend/youjun/pretrained_models/EN_labels_gigaspeech.csv")
key2asr = dict(zip(df['Key'], df['Transcript']))

data = pd.read_csv("/mmu-audio-ssd/frontend/youjun/speechllm/results_emoclap/prediction_asr.csv")

conversational_filler = ['UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER', 'OOF', 'HEE' , 'ACH', 'EEE', 'EW']
unk_tags = ['<UNK>', '<unk>']
gigaspeech_punctuations = ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']
gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
non_scoring_words = conversational_filler + unk_tags + gigaspeech_punctuations + gigaspeech_garbage_utterance_tags
def process_transcription(text):
    # 1. convert to uppercase    
    text = text.upper()
    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace('-', ' ')
    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    remaining_str = ' '.join(remaining_words)
    final_characs = []
    for i in remaining_str:
        if i != ',' and i != '\"' and i != '.' and i != '!':
            final_characs.append(i)
    final_ = "".join(final_characs)
    return final_.lower()

keys, pred_asrs, true_asrs = [], [], []
for i in range(len(data)):
    id = data.iloc[i]['id']
    key = "gigaspeech_"+os.path.basename(id)[:-4]
    asr = data.iloc[i]['asr']
    pred_asr = asr.strip('The content of the speech is ')
    if pred_asr.startswith("'"):
        pred_asr = pred_asr[1:]
    if pred_asr.endswith("'"):
        pred_asr = pred_asr[:-1]

    pred_asrs.append(pred_asr)
    keys.append(key)
    true_asr = key2asr[key]
    true_asr = process_transcription(true_asr)
    true_asrs.append(true_asr)

from jiwer import wer
print(wer(true_asrs, pred_asrs))
