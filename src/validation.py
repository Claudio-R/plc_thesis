import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ValidationDataset
from .utils import resume_from_checkpoint, load_transformer, load_codec

def validation_loop(dataloader, codec, transformer, audio_loss_fn, code_loss_fn):
    transformer.eval()
    running_audio_loss, running_code_loss = (0.0, 0.0)

    progress_bar = tqdm(dataloader, desc='- Validation')
    with torch.no_grad():
        for step, wave24kHz in enumerate(progress_bar):

            wave24kHz = wave24kHz.to(transformer.device)

            # Encode and decode audio_data
            codes = codec.encode(wave24kHz)
            src_codes = codes[..., :-1]
            tgt_codes = codes[..., 1:]

            tgt_audio = codec.decode(tgt_codes)

            # Predict logits
            logits = transformer(src_codes)
            batch_size, n_codebooks, sequence_length, codebook_size = logits.shape

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

            codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)  # shape: (B, n_codebooks, S, C)
            pred_codes = torch.argmax(codebook_index_probs, dim=-1)  # shape: (B, n_codebooks, S), just like input sequence!

            # Predict audio and codes
            pred_audio = codec.decode(pred_codes)

            # Compute predictions and losses
            running_audio_loss += audio_loss_fn(pred_audio, tgt_audio).item()
            running_code_loss += code_loss

            # Update progress bar
            progress_bar.set_postfix({
                'avg_audio_loss': '{0:.5g}'.format(running_audio_loss/(step+1)),
                'avg_code_loss': '{0:.5g}'.format(running_code_loss/(step+1))
            })

        avg_audio_loss = running_audio_loss / len(dataloader)
        avg_code_loss =  running_audio_loss / len(dataloader)

        return avg_audio_loss, avg_code_loss

def validate(config_path):
    print('Validating...')

    with open(config_path) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    version = config['version']
    segment_dur = config['segment_dur']
    device = config['device']

    n_epochs = config['n_epochs']
    steps_per_epoch = config['steps_per_epoch']
    num_workers = config['num_workers']

    validation_metadata_path = config['validation_metadata_path']

    # CODEC
    codec = load_codec('encodec', config).to(device)

    # TRANSFORMER
    transformer = load_transformer(version, config).to(device)
    version, last_epoch = resume_from_checkpoint(transformer, None, version, device) if config["resume"] else (version, 0)

    # DATALOADER
    val_ds = ValidationDataset(codec_sr=codec.sample_rate,
                               metadata_path=validation_metadata_path,
                               data_per_epoch=steps_per_epoch,
                               segment_dur=segment_dur)

    val_loader = DataLoader(val_ds, shuffle=False, num_workers=num_workers)

    # LOSS FUNCTIONS
    code_loss_fn = torch.nn.CrossEntropyLoss()
    audio_loss_fn = torch.nn.L1Loss()

    # TEST LOOP
    for epoch in range(last_epoch, n_epochs + last_epoch):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        validation_loop(val_loader, codec, transformer, audio_loss_fn, code_loss_fn)
    print("Done!")
