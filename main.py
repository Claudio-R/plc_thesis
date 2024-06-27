import sys
import yaml
import warnings

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything

from src.dataset import TrainingDataset, ValidationDataset
from src.lightning_transformer.model import Model

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    try:
        with open('config.yaml') as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)

    segment_dur = config['segment_dur']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    steps_per_epoch = config['steps_per_epoch']
    num_workers = config['num_workers']

    model = Model(config)

    # DATALOADERS
    train_ds = TrainingDataset(codec_sr=model.codec.sample_rate,
                               metadata_path='dataset/vctk/training.csv',
                               data_per_epoch=batch_size * steps_per_epoch,
                               n_epochs=n_epochs,
                               segment_dur=segment_dur)

    val_ds = ValidationDataset(codec_sr=model.codec.sample_rate,
                               metadata_path='dataset/vctk/validation.csv',
                               data_per_epoch=steps_per_epoch,
                               n_epochs=n_epochs,
                               segment_dur=segment_dur)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=num_workers)

    seed_everything(666)
    trainer = Trainer(deterministic=True, strategy='ddp_find_unused_parameters_true')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Done!")



