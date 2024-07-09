import os
import yaml
import warnings
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.dataset import MyDataModule
from src.model import Model
from src.utils import get_mode

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    seed_everything(666)
    with open('config.yaml') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    model_mode = config['model_mode']
    version = config['version']
    mode = get_mode(version)
    dir_ckp = f'{config['dir_ckp']}/{model_mode}/{mode}/{version}/'

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
    model = Model()

    # LOGGER
    logger = TensorBoardLogger(f"meta/tb_logs/{model_mode}", name=f"{mode}", version=version)

    # DATALOADERS
    dm = MyDataModule()

    # TRAINER
    trainer = Trainer(
        strategy='ddp_find_unused_parameters_true',
        callbacks=[checkpoint_callback],
        logger=logger)
    trainer.fit(model, datamodule=dm, ckpt_path='last')
    trainer.test(model, datamodule=dm, ckpt_path='best')
    trainer.predict(model, datamodule=dm, ckpt_path='best')
    print("Done!")



