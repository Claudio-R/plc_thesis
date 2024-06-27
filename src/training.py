import yaml
from torch.utils.data import DataLoader

from .dataset import TrainingDataset, ValidationDataset
from src.lightning_transformer.model import Model
import lightning

def train(config_path):
    print('Training...')

    # CONFIGURATION
    with open(config_path) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    version = config['version']
    segment_dur = config['segment_dur']

    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    steps_per_epoch = config['steps_per_epoch']
    num_workers = config['num_workers']

    speech_dataset = config['speech_dataset']
    environment_dataset = config['environment_dataset']
    music_dataset = config['music_dataset']

    model = Model(config)

    # DATALOADERS
    train_ds = TrainingDataset(codec_sr=model.codec.sample_rate,
                               metadata_path=speech_dataset,
                               data_per_epoch=batch_size*steps_per_epoch,
                               segment_dur=segment_dur)

    val_ds = ValidationDataset(codec_sr=model.codec.sample_rate,
                               metadata_path=speech_dataset,
                               data_per_epoch=steps_per_epoch,
                               segment_dur=segment_dur)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, shuffle=True, num_workers=num_workers)

    trainer = lightning.Trainer()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # # TRAINING LOOP
    # for epoch in epochs:
    #     print(f"\nEpoch {epoch+1}/{epochs[-1]+1}")
    #     training_loop(train_loader, codec, transformer, code_loss_fn, optimizer)
    #     avg_audio_loss, avg_code_loss = validation_loop(val_loader, codec, transformer, audio_loss_fn, code_loss_fn)
    #     if config['save']:
    #         save(version, epoch+1, transformer, optimizer, avg_audio_loss, avg_code_loss, config['epoch_to_save'])
    #     scheduler.step(avg_code_loss)
    print("Done!")
