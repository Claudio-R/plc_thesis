import os
import yaml
import warnings
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.dataset import TrainingDataset, ValidationDataset, TestDataset, PredictDataset
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

    # CHECKPOINT
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_ckp,
        filename='my_model-{epoch:02d}-{accuracy:.5f}',
        monitor='accuracy',
        mode='max',
        save_top_k=5,
        verbose=True
    )

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
        model = Model()

    # LOGGER
    logger = TensorBoardLogger(f"meta/tb_logs", name=f"{mode}", version=version)

    # DATALOADERS
    train_ds = TrainingDataset(codec_sr=model.codec.sample_rate,
                               metadata_path='dataset/vctk/training.csv',
                               data_per_epoch=batch_size * steps_per_epoch,
                               segment_dur=segment_dur)

    val_ds = ValidationDataset(codec_sr=model.codec.sample_rate,
                               metadata_path='dataset/vctk/validation.csv',
                               data_per_epoch=steps_per_epoch,
                               segment_dur=segment_dur)

    test_ds = TestDataset(codec_sr=model.codec.sample_rate,
                          metadata_path='dataset/vctk/test.csv',
                          segment_dur=segment_dur,
                          frame_dim=model.codec.frame_dim)

    pred_ds = PredictDataset(codec_sr=model.codec.sample_rate,
                          metadata_path='dataset/plc_challenge/predict.csv',
                          segment_dur=segment_dur,
                          frame_dim=model.codec.frame_dim)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, shuffle=False, num_workers=num_workers)
    pred_loader = DataLoader(pred_ds, shuffle=False, num_workers=num_workers)

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))
    print(len(pred_loader))

    input('...')

    trainer = Trainer(
        strategy='ddp_find_unused_parameters_true',
        callbacks=[checkpoint_callback],
        logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader, ckpt_path='best')
    print("Done!")



