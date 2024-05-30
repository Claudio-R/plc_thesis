import os
import yaml
import torch
from torch.utils.data import DataLoader
import soundfile as sf
from tqdm import tqdm

from .v1 import PLCModel_v1
from .v2 import PLCModel_v2
from .utils import create_trace, simulate_packet_loss, load_model, resume_from_checkpoint
from .dataset import TestDataset


def test_loop(tests_dir, test_ID, dataloader, model, version, sr=44100):
    
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
    model.eval()
    with torch.no_grad():
        for Track_ID, (audio_test_path, audio_data, trace) in enumerate(tqdm(dataloader, desc='- Testing')):
            audio_test_path = audio_test_path[0]
            audio_data = audio_data[0]
            trace = trace[0].to('cpu').numpy()

            # Generate target signals
            codes = model.encode(audio_data)
            tgt_audio = model.decode(codes)
            tgt_audio_lost = simulate_packet_loss(tgt_audio, trace)

            # Inference
            model_audio_pred = model.inference(tgt_audio, tgt_audio_lost, trace)

            # Save audio files
            sf.write(f'{clean_dir}/tgt_audio_{Track_ID}.wav', tgt_audio.squeeze().to('cpu'), sr)
            sf.write(f'{lossy_dir}/tgt_audio_{Track_ID}.wav', tgt_audio_lost.squeeze().to('cpu'), sr)
            sf.write(f'{enhanced_dir}/tgt_audio_{Track_ID}.wav', model_audio_pred.squeeze().to('cpu'), sr)
            
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

    model = load_model(version, config)
    version, _ = resume_from_checkpoint(model, None, version) if resume else (version, 0)

    test_ds = TestDataset(sample_rate=model.sample_rate,
                          metadata_path=test_metadata_path,
                          data_per_epoch=steps_per_epoch,
                          segment_dur=segment_dur)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    
    test_loop(tests_dir, 2, test_loader, model, version)
