import os
import yaml
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from src import utils
from tqdm import tqdm
from src.dataset import TestDataset
from torch.utils.data import DataLoader

# Pick a track from dataset, plc is okay
# Read first 0.5 second
# Generate audio samples up to 5 seconds (so that last packets will be predicted from newly generated ones only)
# Repeat reading first 1 second and 2 seconds
# Repeat picking another clean track from dataset

SR = None
DURATIONS = [0.5, 1, 2]
TOTAL_DURATION = 5
CONFIG_PATH = 'config.yaml'
DESTINATION_DIR = 'tests/generate_tracks'

def generate_script():

    with open(CONFIG_PATH) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    version = config['version']
    device = config['device']
    frame_dim = config["frame_dim"]
    segment_dur = config["segment_dur"]
    num_workers = config["num_workers"]
    resume = config['resume']
    test_metadata_path = config['plc_challenge_path']

    # CODEC
    codec = utils.load_codec('encodec', config).to(device)

    print(codec.codebook_size)
    print(codec.n_codebooks)
    input(',,,')

    sr = codec.sample_rate
    total_num_frames = int((sr * TOTAL_DURATION) / frame_dim)

    # TRANSFORMER
    transformer = utils.load_transformer(version, config).to(device)
    version, _ = utils.resume_from_checkpoint(transformer, None, version, device) if resume else (version, 0)
    max_sequence_length = int(transformer.context_length / frame_dim)
    transformer.eval()

    # DATALOADER
    test_ds = TestDataset(codec_sr=codec.sample_rate,
                          metadata_path=test_metadata_path,
                          segment_dur=segment_dur,
                          frame_dim=frame_dim)

    dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    for Track_ID, (wave, _) in enumerate(tqdm(dataloader, desc='- Generating')):

        wave = wave.to(device)
        for dur in DURATIONS:

            pred_dir = f'{DESTINATION_DIR}/{version}/{dur}_seconds/'
            os.makedirs(pred_dir, exist_ok=True)

            num_samples = int((sr * dur)/frame_dim) * frame_dim
            wave_segment = wave[..., :num_samples]
            codes = codec.encode(wave_segment)

            while codes.shape[-1] <= total_num_frames:
                curr_seq_length = codes.shape[-1]
                first_frame = max(0, curr_seq_length - max_sequence_length)
                src_codes = codes[..., first_frame:]
                logits = transformer(src_codes)
                codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_codes = torch.argmax(codebook_index_probs, dim=-1)
                # print(f'predicted_code: {pred_codes[...,-1]}')
                # input('Enter to proceed')

                codes = torch.cat((codes, pred_codes[..., -1].unsqueeze(-1)), dim=-1)
                # print(f"codes: {codes[...,-1]}")

            pred = codec.decode(codes)
            sf.write(pred_dir+f'generated_{Track_ID}.wav', pred.squeeze().to('cpu'), sr)







