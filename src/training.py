import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .validation import validation_loop
from .dataset import TrainingDataset, ValidationDataset
from .utils import resume_from_checkpoint, save, load_codec, load_transformer

def training_loop(dataloader, codec, transformer, code_loss_fn, optimizer):
    transformer.train()
    running_code_loss = 0.0

    progress_bar = tqdm(dataloader, desc='- Training')
    for step, wave24kHz in enumerate(progress_bar):
        # (B, 1, T), (B, 1, T/2)

        wave24kHz = wave24kHz.to(transformer.device)

        # Encode audio_data: 24000 samples --> 75 packets: 1 packet --> 13 ms
        codes = codec.encode(wave24kHz) #(64, 8, 150)

        # PARALLEL PATTERN
        src_codes = codes[..., :-1]
        tgt_codes = codes[..., 1:]

        # DELAYED PATTERN
        # # remove last 4 tokens... so you may want to start from (B, 4, 154), or not... who cares!
        # # EDIT: 06.20.24 per il momento utilizzo sequence di 150 e ne tolgo 4
        # nq = transformer.n_codebooks
        # src_codes = codes[..., :-nq]
        # # assuming codes[:, 0, :] to be the first quantizer, i.e., the most important
        # for i in range(1, nq):
        #     codes[:, i:, :] = torch.roll(codes[:, i:, :], shifts=1, dims=-1)
        # # remove first 4 tokens... so you may want to start from (B, 4, 154), or not... who cares!
        # tgt_codes = codes[..., nq:]

        # Call transformer
        logits = transformer(src_codes)
        batch_size, n_codebooks, sequence_length, codebook_size = logits.shape

        # Compute loss
        code_loss = 0.0
        for k in range(n_codebooks):
            # Extract logits and targets for the current codebook
            logits_k = logits[:, k, :, :].contiguous()  # Shape: [batch_size, sequence_length, codebook_size]
            targets_k = tgt_codes[:, k, :].contiguous()  # Shape: [batch_size, sequence_length]

            logits_k = logits_k.view(-1, logits_k.size(-1))  # Shape: [batch_size * sequence_length, codebook_size]
            targets_k = targets_k.view(-1)  # Shape: [batch_size * sequence_length]

            code_loss_k = code_loss_fn(logits_k, targets_k)
            code_loss += code_loss_k

        code_loss /= n_codebooks

        # Backpropagation
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

    # CONFIGURATION
    with open(config_path) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    version = config['version']
    segment_dur = config['segment_dur']
    device = config['device']

    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    steps_per_epoch = config['steps_per_epoch']
    learning_rate = config['learning_rate']
    num_workers = config['num_workers']
    kbps = config['codec']['bitrate']

    speech_dataset = config['speech_dataset']
    environment_dataset = config['environment_dataset']
    music_dataset = config['music_dataset']

    # CODEC
    codec = load_codec('encodec', kbps).to(device)

    # TRANSFORMER
    transformer = load_transformer(version, config).to(device)

    # OPTIMIZER
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
    version, last_epoch = resume_from_checkpoint(transformer, optimizer, version, device) if config["resume"] else (version, 0)
    epochs = range(last_epoch, n_epochs+last_epoch)

    # SCHEDULER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # DATALOADERS
    train_ds = TrainingDataset(codec_sr=codec.sample_rate,
                               metadata_path=speech_dataset,
                               data_per_epoch=batch_size*steps_per_epoch,
                               segment_dur=segment_dur)

    val_ds = ValidationDataset(codec_sr=codec.sample_rate,
                               metadata_path=speech_dataset,
                               data_per_epoch=steps_per_epoch,
                               segment_dur=segment_dur)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, shuffle=True, num_workers=num_workers)

    # LOSS FUNCTIONS
    code_loss_fn = torch.nn.CrossEntropyLoss()
    audio_loss_fn = torch.nn.L1Loss()

    # TRAINING LOOP
    for epoch in epochs:
        print(f"\nEpoch {epoch+1}/{epochs[-1]+1}")
        training_loop(train_loader, codec, transformer, code_loss_fn, optimizer)
        avg_audio_loss, avg_code_loss = validation_loop(val_loader, codec, transformer, audio_loss_fn, code_loss_fn)
        if config['save']:
            save(version, epoch+1, transformer, optimizer, avg_audio_loss, avg_code_loss, config['epoch_to_save'])
        scheduler.step(avg_code_loss)

    print("Done!")
