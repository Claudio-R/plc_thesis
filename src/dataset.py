import os
import random
from typing import Union
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import tqdm
from audiomentations import Compose, AddGaussianNoise, PolarityInversion, PitchShift, TanhDistortion, TimeStretch
import torch
# import musdb
# from datasets import load_dataset
import soundfile as sf
from .utils import create_trace
import math
# import torchaudio as ta
# from torchaudio.transforms import Resample
import librosa

def load_from_dataset(subset: str,
                      music_dataset: str = "/nas/home/crapisarda/Medley-solo-DB",
                      speech_dataset: str = "/nas/public/dataset/VCTK/wav48_silence_trimmed") -> pd.DataFrame:

    handle = {"type": [], "path": [], "duration": []}

    # Music Dataset
    music_csv = pd.read_csv(os.path.join(music_dataset, 'Medley-solos-DB_metadata.csv'))
    dataset = [row for i, row in music_csv.iterrows() if row['subset'] == subset]

    for row in tqdm.tqdm(dataset, desc=f"Collecting music tracks for {subset}"):
        SUBSET = row['subset']
        INSTRUMENTID = row['instrument_id']
        UUID = row['uuid4']
        path = os.path.join(music_dataset, f'dataset/Medley-solos-DB_{SUBSET}-{INSTRUMENTID}_{UUID}.wav')
        data, sample_rate = sf.read(path)
        handle["type"].append("music")
        handle["path"].append(path)
        handle["duration"].append(len(data) / sample_rate)

    # Speech dataset
    speakers = [os.path.join(speech_dataset, x) for x in os.listdir(speech_dataset) if
                os.path.isdir(os.path.join(speech_dataset, x))]

    tracks_path = []
    for x in speakers:
        tracks_path = tracks_path + [os.path.join(x, track) for track in os.listdir(x) if
                                     track.lower().endswith('.flac')]

    split = 0.3
    if subset == 'training':
        tracks_path = tracks_path[0:int(split * len(tracks_path))]
    elif subset == 'validation':
        tracks_path = tracks_path[int(split * len(tracks_path)):int(2 * split * len(tracks_path))]
    elif subset == 'test':
        tracks_path = tracks_path[int(2 * split * len(tracks_path)):]

    for track in tqdm.tqdm(tracks_path, desc="Collecting speech tracks for training"):
        handle["type"].append("speech")
        handle["path"].append(track)
        data, sample_rate = sf.read(track)
        handle["duration"].append(len(data) / sample_rate)

    print('Dataset size: ', len(handle["type"]))
    df = pd.DataFrame(handle)
    return df


def load_random_audio_segment(df: pd.DataFrame, codec_sr: int, segment_dur: float,
                              p: list = [0.3, 0.3, 0.3, 0.3, 0.3], augment_bool:bool = False) -> torch.Tensor:
    """
    Load a random audio file according to probability and applies audio augmentation
    :returns:
    audio_data: Tensor of shape (1, segment_dur*sample_rate)
    """

    # audio_type = "speech" if torch.rand(1) > 0.5 else "music"
    audio_type = "speech"
    sample = df.loc[((df.type == audio_type) & (df.duration >= segment_dur))].sample()

    dur = sample["duration"].values[0]
    path = sample["path"].values[0]
    random_offset = random.uniform(0, dur - (segment_dur))

    # Applies resample
    # BUG: ta.load carica frammenti di lunghezza random
    wave24kHz, sr = librosa.load(path, sr=codec_sr, offset=random_offset, duration=(segment_dur), mono=True)

    if augment_bool:
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p[0]),
            PolarityInversion(p=p[1]),
            PitchShift(min_semitones=-4, max_semitones=4, p=p[2]),
            TanhDistortion(p=p[3]),
            TimeStretch(min_rate=1, max_rate=1.25, p=p[4]),
        ])
        wave24kHz = augment(wave24kHz, sample_rate=codec_sr)

    wave24kHz = wave24kHz[np.newaxis, :int((segment_dur) * codec_sr)]

    return wave24kHz


def load_random_audio_mix(df: pd.DataFrame, codec_sr: int, segment_dur: float,
                          alpha: float = 2.0,
                          mix_bool:bool = False,
                          augment_bool:bool = False):

    wave24kHz = load_random_audio_segment(df, codec_sr, segment_dur, augment_bool=augment_bool)
    if mix_bool:
        for _ in range(2):
            gain = np.random.beta(alpha, alpha)
            signal_24kHz_ = load_random_audio_segment(df, codec_sr, segment_dur, augment_bool=augment_bool)
            wave24kHz = gain * wave24kHz + (1 - gain) * signal_24kHz_

    return wave24kHz


class TrainingDataset(Dataset):
    def __init__(self, *,
                 codec_sr: int,
                 metadata_path: str,
                 data_per_epoch: int,
                 segment_dur: float
                 ):
        self.codec_sr = codec_sr
        self.data_per_epoch = data_per_epoch
        self.segment_dur = segment_dur
        self.metadata = self.load_dataframe(metadata_path)

    def load_dataframe(self, metadata_path) -> pd.DataFrame:
        with open(metadata_path, 'rb') as handle:
            print(f'Loading data from cache: {metadata_path}.')
            df = pd.read_csv(handle)
        try:
            pass
        except:
            print(f'Cannot locate caches at {metadata_path}. Collecting data...')
            df = load_from_dataset(subset='training')
            df.to_csv(metadata_path, index=False)
        return df

    def __len__(self):
        return self.data_per_epoch

    @torch.no_grad()
    def __getitem__(self, index) -> torch.Tensor:
        """
        :param index:
        :return: wave_48kHz, wave_24kHz
        """
        return load_random_audio_mix(self.metadata, self.codec_sr, self.segment_dur, mix_bool=True, augment_bool=True)


class ValidationDataset(Dataset):
    def __init__(self, *,
                 codec_sr: int,
                 metadata_path: str,
                 data_per_epoch: int,
                 segment_dur: float,
                 ):
        self.codec_sr = codec_sr
        self.segment_dur = segment_dur
        self.data_per_epoch = data_per_epoch
        self.metadata = self.load_dataframe(metadata_path)

    def __len__(self):
        return self.data_per_epoch

    def load_dataframe(self, metadata_path) -> pd.DataFrame:
        try:
            with open(metadata_path, 'rb') as handle:
                print(f'Loading data from cache: {metadata_path}.')
                df = pd.read_csv(handle)

        except:
            print(f'Cannot locate caches at {metadata_path}. Collecting data...')
            df = load_from_dataset(subset='validation')
            df.to_csv(metadata_path, index=False)

        return df

    @torch.no_grad()
    def __getitem__(self, index):
        return load_random_audio_mix(self.metadata, self.codec_sr, self.segment_dur, mix_bool=True, augment_bool=True)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, *,
                 codec_sr: int,
                 metadata_path: str,
                 segment_dur: float,
                 frame_dim: int,
                 use_random_trace:bool = False):
        self.codec_sr = codec_sr
        self.segment_dur = segment_dur
        self.metadata = self.load_dataframe(metadata_path)
        self.frame_dim = frame_dim
        self.use_random_trace = use_random_trace

    def load_dataframe(self, metadata_path) -> pd.DataFrame:
        try:
            with open(metadata_path, 'rb') as handle:
                print(f'Loading data from cache: {metadata_path}.')
                df = pd.read_csv(handle)
        except Exception as e:
            print(f'Cannot locate caches at {metadata_path}. Collecting data...')
            print(e)
            df = load_from_dataset(subset='test')
            filtered_speech = df[df['type'] == 'speech'].sample(n=10, random_state=42)
            filtered_music = df[df['type'] == 'music'].sample(n=10, random_state=42)

            # Concatenate the sampled rows
            df = pd.concat([filtered_speech, filtered_music], ignore_index=True)
            df.to_csv(metadata_path, index=False)

        return df

    def __len__(self):
        return self.metadata.shape[0]

    @torch.no_grad()
    def __getitem__(self, index):
        sample = self.metadata.loc[index]
        wave24kHz, sr = librosa.load(sample.path, sr=self.codec_sr, mono=True)
        wave24kHz = wave24kHz[np.newaxis, :]

        # Adapt PLC Challenge traces to new samplerate
        if 'trace' in self.metadata.columns:
            trace = sample.trace.split()
            num_packets = math.ceil(wave24kHz.shape[-1] // self.frame_dim)
            pad_length = num_packets - len(trace)
            for i in range(pad_length):
                trace.append(trace[i])
            assert(len(trace) == num_packets)
            trace = np.array([int(i) for i in trace])

        # Or create new traces
        else:
            trace = create_trace(wave24kHz, self.frame_dim, random_trace=self.use_random_trace)

        return wave24kHz, trace

