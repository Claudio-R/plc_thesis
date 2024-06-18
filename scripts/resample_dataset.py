import os
from pathlib import Path
import librosa
import soundfile as sf
import rglob
import pandas as pd

# Resample Train, Val, Test, PLC
DATASET_DIR = "/nas/public/dataset/VCTK/wav48_silence_trimmed"
DESTINATION_DIR = 'dataset'
DESTINATION_DIR = '../'
NEW_SR = 24000

if __name__ == '__main__':
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    wavs = sorted(Path(DATASET_DIR).rglob('*.wav'))
    for wav in wavs:
        y, sr = librosa.load(wav, sr=NEW_SR)
        # TODO: fix here
        sf.write(os.path.join(DESTINATION_DIR, wav.split('/')[-1]), y, sr)
