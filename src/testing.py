import os

import librosa
import yaml
import torch
from torch.utils.data import DataLoader
import soundfile as sf
from tqdm import tqdm

from .utils import simulate_packet_loss, load_transformer, load_codec, resume_from_checkpoint
from .dataset import TestDataset

def test_loop(tests_dir, test_ID, dataloader, codec, transformer, version, sr=44100):
    
    # Create folders
    test_dir = f'{tests_dir}/t{test_ID}'
    clean_dir = f'{test_dir}/clean'
    lossy_dir = f'{test_dir}/lossy'
    version_id = version.split('.')[0]
    enhanced_dir = f'{test_dir}/enhanced/model_v{version_id}/{version}'
    traces_dir = f'{test_dir}/traces'

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(lossy_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    # Start testing
    transformer.eval()

    with torch.no_grad():
        for Track_ID, (wave24kHz, trace) in enumerate(tqdm(dataloader, desc='- Testing')):

            wave24kHz = wave24kHz.to(codec.device)

            # Encode and decode audio_data
            codes = codec.encode(wave24kHz)
            tgt_wave24kHz = codec.decode(codes)

            # Simulate_packet_loss
            tgt_wave24kHz_lost = simulate_packet_loss(tgt_wave24kHz, trace, packet_dim=codec.frame_dim)


            # Inference
            codes_lost = codec.encode(tgt_wave24kHz_lost)
            sequence_length = int(transformer.context_length / codec.frame_dim)
            for i, loss in enumerate(trace):
                if loss:
                    first_idx = max(0, i - sequence_length)
                    src_codes = codes_lost[..., first_idx:i]
                    logits = transformer(src_codes)
                    codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)
                    pred_codes = torch.argmax(codebook_index_probs, dim=-1)
                    codes_lost[..., i] = pred_codes[..., -1]

            pred_wave24kHz = codec.decode(codes_lost)

            # Save audio files
            sf.write(f'{clean_dir}/tgt_audio_{Track_ID}.wav', tgt_wave24kHz.squeeze().to('cpu'), sr)
            sf.write(f'{lossy_dir}/tgt_audio_{Track_ID}.wav', tgt_wave24kHz_lost.squeeze(), sr)
            sf.write(f'{enhanced_dir}/tgt_audio_{Track_ID}.wav', pred_wave24kHz.squeeze().to('cpu'), sr)
            
            # Save traces
            with open(f'{traces_dir}/tgt_audio_{Track_ID}.txt', 'w') as f:
                for trace_idx in trace:
                    f.write('{}\n'.format(trace_idx))

def test(config_path, resume=True):
    
    print('Testing...')
    
    with open(config_path) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    steps_per_epoch = config['steps_per_epoch']
    test_metadata_path = config['plc_challenge_path']
    segment_dur = config['segment_dur']
    num_workers = config['num_workers']
    tests_dir = config['tests_dir']
    version = config['version']
    transformer_device = config["transformer"]['device']
    codec_device = config["codec"]['device']

    # CODEC
    codec = load_codec('encodec', config).to(codec_device)

    # TRANSFORMER
    transformer = load_transformer(version, config).to(transformer_device)

    version, _ = resume_from_checkpoint(transformer, None, version) if resume else (version, 0)

    test_ds = TestDataset(codec_sr=codec.sample_rate,
                          metadata_path=test_metadata_path,
                          data_per_epoch=steps_per_epoch,
                          segment_dur=segment_dur)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    
    test_loop(tests_dir, 2, test_loader, codec, transformer, version)
