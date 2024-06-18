import os
import yaml
import soundfile as sf
from src import utils
from tqdm import tqdm
from src.dataset import TestDataset
from torch.utils.data import DataLoader

KBPS = [1.5, 3., 6., 12.]
CONFIG_PATH = 'config.yaml'
CODEC_NAME = 'encodec'
DESTINATION_DIR = f'tests/test_codec/{CODEC_NAME}'

def test_codec():

    with open(CONFIG_PATH) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    device = config['device']
    frame_dim = config["frame_dim"]
    segment_dur = config["segment_dur"]
    num_workers = config["num_workers"]
    test_metadata_path = config['plc_challenge_path']

    # CODEC
    sr = 24000

    # DATALOADER
    test_ds = TestDataset(codec_sr=sr,
                          metadata_path=test_metadata_path,
                          segment_dur=segment_dur,
                          frame_dim=frame_dim)

    dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    for Track_ID, (wave, _) in enumerate(tqdm(dataloader, desc='- Testing Codec')):
        wave = wave.to(device)
        for k in KBPS:
            codec = utils.load_codec(CODEC_NAME, k).to(device)
            pred_dir = f'{DESTINATION_DIR}/kbps_{k}/'
            os.makedirs(pred_dir, exist_ok=True)
            codes = codec.encode(wave)
            pred = codec.decode(codes)
            sf.write(pred_dir+f'decoded_{Track_ID}.wav', pred.squeeze().to('cpu'), sr)







