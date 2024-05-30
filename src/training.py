import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.spatial import distance
from scipy.io import wavfile

from .v1.PLCModel import PLCModel as PLCModel_v1
from .v2.PLCModel import PLCModel as PLCModel_v2
from .validation import validation_loop
from .testing import test_loop
from .dataset import TrainingDataset, ValidationDataset
from .utils import resume_from_checkpoint, save, load_model


def training_loop(epoch, dataloader, model, code_loss_fn, optimizer, device, version='1.0.0'):
    model.train()

    running_code_loss = 0.0

    progress_bar = tqdm(dataloader, desc='- Training')
    for step, audio_data in enumerate(progress_bar):

        # Encode audio_data
        codes = model.encode(audio_data)

        src_codes = codes[..., :-1]
        tgt_codes = codes[..., 1:]

        # Predict logits
        logits = model(src_codes)
        batch_size, n_codebooks, sequence_length, codebook_size = logits.shape

        code_loss = 0.0  # inizializza la tua (average) cross-entropy (ce) loss

        for k in range(n_codebooks):
            # Extract logits and targets for the current codebook
            logits_k = logits[:, k, :, :].contiguous()  # Shape: [batch_size, sequence_length, codebook_size]
            targets_k = tgt_codes[:, k, :].contiguous()  # Shape: [batch_size, sequence_length]

            logits_k = logits_k.view(-1, logits_k.size(-1))  # Shape: [batch_size * sequence_length, codebook_size]
            targets_k = targets_k.view(-1)  # Shape: [batch_size * sequence_length]

            code_loss_k = code_loss_fn(logits_k, targets_k)
            # if needed, e.g., for debug purposes, consider logging ce_k.detach() here

            code_loss += code_loss_k

        code_loss /= n_codebooks

        # Backpropagate
        code_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # # Save loss history
        code_loss = code_loss.item()
        running_code_loss += code_loss

        # Update progress bar
        progress_bar.set_postfix({
            'code_loss': '{0:.5g}'.format(code_loss),
            'avg_code_loss': '{0:.5g}'.format(running_code_loss / (step + 1))
        })


def train(config_path):
    print('Training...')

    with open(config_path) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    steps_per_epoch = config['steps_per_epoch']
    learning_rate = config['learning_rate']
    training_metadata_path = config['training_metadata_path']
    validation_metadata_path = config['validation_metadata_path']
    segment_dur = config['segment_dur']
    num_workers = config['num_workers']
    pin_memory = config['pin_memory']
    transformer_device = config["transformer"]['device']
    dac_device = config["dac"]['device']
    version = config['version']

    model = load_model(version, config)
    optimizer = torch.optim.Adam(model.transformer.parameters(), lr=learning_rate)

    version, last_epoch = resume_from_checkpoint(model, optimizer, version) if config["resume"] else (version, 0)
    epochs = range(last_epoch, n_epochs+last_epoch)

    train_ds = TrainingDataset(sample_rate=model.sample_rate,
                               metadata_path=training_metadata_path,
                               data_per_epoch=batch_size*steps_per_epoch,
                               segment_dur=segment_dur,
                               device=transformer_device)

    val_ds = ValidationDataset(sample_rate=model.sample_rate,
                               metadata_path=validation_metadata_path,
                               data_per_epoch=steps_per_epoch,
                               segment_dur=segment_dur,
                               device=transformer_device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    code_loss_fn = torch.nn.CrossEntropyLoss()
    audio_loss_fn = torch.nn.L1Loss()

    for epoch in epochs:
        print(f"\nEpoch {epoch+1}/{epochs[-1]+1}")
        training_loop(epoch, train_loader, model, code_loss_fn, optimizer, transformer_device)
        avg_audio_loss, avg_code_loss = validation_loop(epoch, val_loader, model, audio_loss_fn, code_loss_fn, transformer_device)
        if config['save']: save(version, epoch+1, model, optimizer, avg_audio_loss, avg_code_loss, config['epoch_to_save'])

    print("Done!")
