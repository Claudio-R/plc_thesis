import os
import yaml
import warnings
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.dataset import MyDataModule
from src.model import Model
from src.utils import get_mode

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    seed_everything(666)
    with open('config.yaml') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    model_mode = config['model_mode']
    version = config['version']
    mode = get_mode(version)
    dir_ckp = f'{config['dir_ckp']}/{model_mode}/{mode}/{version}/'

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
    logger = TensorBoardLogger(f"meta/tb_logs/{model_mode}", name=f"{mode}", version=version)

    dm = MyDataModule()

    trainer = Trainer(
        strategy='ddp_find_unused_parameters_true',
        logger=logger)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)
    print("Done!")