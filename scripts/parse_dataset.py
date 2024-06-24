import os, glob
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

def parse_dataset():
    FORMATS = ['wav', 'flac', 'mp3']
    DATASET = 'jamendo'
    SR = 24000
    path = f'dataset/{DATASET}'
    csv_path = f'dataset/{DATASET}.csv'

    handle = {"type": [], "path": [], "duration": []}
    files = []
    for filename in glob.iglob(path + '**/**', recursive=True):
        ext = filename.split('.')[-1]
        if ext in FORMATS:
            files.append(filename)

    os.makedirs(f'{path}/sr_{SR}Hz', exist_ok=True)
    for filename in tqdm(files, f'Resampling files in {DATASET}', total=len(files)):
        y_res, _ = librosa.load(filename, sr=SR)
        duration = y_res.shape[-1] / SR
        new_path = f'{path}/sr_{SR}Hz/{filename.split('/')[-1]}'
        sf.write(new_path, y_res, SR)
        handle["type"].append('speech')
        handle["path"].append(new_path)
        handle["duration"].append(duration)

    print('Dataset size: ', len(handle["type"]))
    df = pd.DataFrame(handle)
    df.to_csv(csv_path, index=False)
