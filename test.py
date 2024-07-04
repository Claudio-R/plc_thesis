import os
import yaml
import warnings
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.dataset import TestDataset
from src.model.model import Model

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    seed_everything(666)
    with open('config.yaml') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    segment_dur = config['segment_dur']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    steps_per_epoch = config['steps_per_epoch']
    num_workers = config['num_workers']
    mode = config['mode']
    version = config['version']
    dir_ckp = f'{config['dir_ckp']}/{mode}/{version}/'

    # MODEL
    if os.path.exists(dir_ckp):
        best_acc = 0.0
        best_ckp = None
        for f in os.listdir(dir_ckp):
            accuracy = float(f.split('=')[-1].split('.ckpt')[0])
            if accuracy > best_acc:
                best_acc = accuracy
                best_ckp = os.path.join(dir_ckp, f)
        print('Loading {}'.format(best_ckp))
        model = Model.load_from_checkpoint(best_ckp)
    else:
        raise FileNotFoundError('No checkpoint found at {}'.format(dir_ckp))

    # LOGGER
    logger = TensorBoardLogger(f"meta/tb_logs", name=f"{mode}_model", version=version)

    # DATALOADERS
    test_ds = TestDataset(codec_sr=model.codec.sample_rate,
                          metadata_path='dataset/plc_challenge/test.csv',
                          segment_dur=segment_dur,
                          frame_dim=model.codec.frame_dim)

    test_loader = DataLoader(test_ds, shuffle=False)

    trainer = Trainer(
        strategy='ddp_find_unused_parameters_true',
        logger=logger)
    trainer.test(model, test_loader)
    print("Done!")



