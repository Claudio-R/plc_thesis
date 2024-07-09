import os, yaml
import random
from typing import Union
import numpy as np
import pandas as pd
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import tqdm
from audiomentations import Compose, AddGaussianNoise, PolarityInversion, PitchShift, TanhDistortion, TimeStretch
import torch
import soundfile as sf
# from .utils import create_trace
import math
import librosa
import lightning as L
from src.codecs.encodec24kHz import EnCodec24kHz

class MyDataModule(L.LightningDataModule):
    def __init__(self,
                 train_csv:str='dataset/vctk/training.csv', #(46062, 3)
                 val_csv:str='dataset/vctk/validation.csv', #(24302, 3)
                 test_csv:str='dataset/vctk/test.csv', #(20624, 3)
                 predict_csv:str='dataset/plc_challenge/predict.csv', #(20, 2)
                 ):

        super().__init__()
        with open('config.yaml') as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)

        codec_sr = config['codec']['sample_rate']
        segment_dur = config['segment_dur']
        frame_dim = config['frame_dim']
        num_workers = config['num_workers']
        batch_size = config['batch_size']

        self.train_loader = DataLoader(
            MyDataset(train_csv, codec_sr, segment_dur),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.val_loader = DataLoader(
            MyDataset(val_csv, codec_sr, segment_dur),
            shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(
            MyDataset(test_csv, codec_sr, segment_dur),
            shuffle=False, num_workers=num_workers)
        self.predict_loader = DataLoader(
            PredictDataset(predict_csv, codec_sr, segment_dur, frame_dim),
            shuffle=False, num_workers=num_workers)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.predict_loader

class MyDataset(Dataset):
    def __init__(self,
                 csv_path:str,
                 codec_sr:int=24000,
                 segment_dur:float=2.0,
                 ) -> Dataset:
        self.csv_path = csv_path
        self.codec_sr = codec_sr
        self.segment_dur = segment_dur
        self.data = self.load_dataframe()

    def __len__(self) -> int:
        return self.data.shape[0]

    @torch.no_grad
    def __getitem__(self, idx) -> torch.Tensor:
        audio = self.load_audio_segment(idx)
        audio = self.augment(audio)
        audio = self.mix(audio)
        audio = torch.Tensor(audio).unsqueeze(0)
        return audio

    def load_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path)
            df = df.loc[df['duration'] >= self.segment_dur]
            return df
        except Exception as e:
            raise e

    def mix(self, audio, n: int = 3, gain: float = 0.2) -> np.ndarray:
        for i in range(n):
            gain = gain / (i + 1)
            new_audio = self.load_audio_segment()
            audio = audio + gain * new_audio
        return audio

    def augment(self, audio, p: list = [0.3, 0.3, 0.3, 0.3, 0.3]) -> np.ndarray:
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p[0]),
            PolarityInversion(p=p[1]),
            PitchShift(min_semitones=-4, max_semitones=4, p=p[2]),
            TanhDistortion(p=p[3]),
            TimeStretch(min_rate=1, max_rate=1.25, p=p[4]),
        ])
        num_samples = len(audio)
        audio = augment(audio, sample_rate=self.codec_sr)
        audio = audio[:num_samples]
        return audio

    def load_audio_segment(self, idx:int=None) -> np.ndarray:
        if idx:
            sample = self.data.iloc[idx]
            path = sample['path']
            dur = sample['duration']
        else:
            sample = self.data.sample(1)
            path = sample["path"].values[0]
            dur = sample["duration"].values[0]
        random_offset = random.uniform(0, dur - (self.segment_dur))
        audio, sr = librosa.load(path, sr=self.codec_sr, offset=random_offset, duration=self.segment_dur, mono=True)
        return audio

class PredictDataset(MyDataset):
    def __init__(self,
                 csv_path:str,
                 codec_sr:int=24000,
                 segment_dur:float=2.0,
                 frame_dim:int=320
                 ):

        super().__init__(csv_path, codec_sr, segment_dur)
        self.frame_dim = frame_dim

    @torch.no_grad
    def __getitem__(self, index):
        sample = self.data.loc[index]
        audio, sr = librosa.load(sample.path, sr=self.codec_sr, mono=True)
        audio = audio[np.newaxis, :]

        # Adapt PLC Challenge traces to new samplerate of 24kHz
        if 'trace' in self.data.columns:
            trace = sample.trace.split()
            num_packets = math.ceil(audio.shape[-1] // self.frame_dim)
            pad_length = num_packets - len(trace)
            for i in range(pad_length):
                trace.append(trace[i])
            assert (len(trace) == num_packets)
            trace = np.array([int(i) for i in trace])
        else:
            trace = self.create_trace(audio, self.frame_dim)
        return audio, trace

    def load_dataframe(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.csv_path)
        except Exception as e:
            raise e

    def create_trace(self, audio, random_trace:bool = True, loss_prob:float=0.4, loss_rate: int=10) -> np.ndarray:
        trace_len = math.ceil(audio.shape[-1] // self.frame_dim)
        trace = np.zeros(trace_len, dtype=int)
        if not random_trace:
            trace[np.arange(loss_rate, trace_len, loss_rate)] = 1
        else:
            for idx in range(1, trace_len):
                trace[idx] = 0 if random.uniform(0, 1) > loss_prob else 1
        return trace