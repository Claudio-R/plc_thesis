import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .validation import validation_loop
from .testing import test_loop
from .dataset import TrainingDataset, ValidationDataset
from .utils import resume_from_checkpoint, save, load_codec, load_transformer

def training_loop(dataloader, codec, transformer, code_loss_fn, optimizer):
    transformer.train()
    running_code_loss = 0.0

    progress_bar = tqdm(dataloader, desc='- Training')
    for step, wave24kHz in enumerate(progress_bar):
        # (B, 1, T), (B, 1, T/2)

        wave24kHz = wave24kHz.to(codec.device)

        # Encode audio_data: 24000 samples --> 75 packets: 1 packet --> 13 ms
        codes = codec.encode(wave24kHz) #(64, 8, 150)
        src_codes = codes[..., :-1]
        tgt_codes = codes[..., 1:]

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
    transformer_device = config["transformer"]['device']
    codec_device = config["codec"]['device']
    version = config['version']

    # CODEC
    codec = load_codec('encodec', config).to(codec_device)

    # TRANSFORMER
    transformer = load_transformer(version, config).to(transformer_device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
    version, last_epoch = resume_from_checkpoint(transformer, optimizer, version) if config["resume"] else (version, 0)
    epochs = range(last_epoch, n_epochs+last_epoch)

    train_ds = TrainingDataset(codec_sr=codec.sample_rate,
                               transformer_sr=transformer.sample_rate,
                               metadata_path=training_metadata_path,
                               data_per_epoch=batch_size*steps_per_epoch,
                               segment_dur=segment_dur,
                               device=transformer_device)

    val_ds = ValidationDataset(codec_sr=codec.sample_rate,
                               transformer_sr=transformer.sample_rate,
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
        training_loop(train_loader, codec, transformer, code_loss_fn, optimizer)
        avg_audio_loss, avg_code_loss = validation_loop(val_loader, codec, transformer, audio_loss_fn, code_loss_fn)
        if config['save']:
            save(version, epoch+1, transformer, optimizer, avg_audio_loss, avg_code_loss, config['epoch_to_save'])

    print("Done!")
