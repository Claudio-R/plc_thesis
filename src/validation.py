import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .v1 import PLCModel_v1
from .v2 import PLCModel_v2
from .dataset import ValidationDataset
from .utils import resume_from_checkpoint, load_model, save

def validation_loop(epoch, dataloader, model, audio_loss_fn, code_loss_fn, device, version='1.0.0'):
    model.eval()
    running_audio_loss, running_code_loss = (0.0, 0.0)

    progress_bar = tqdm(dataloader, desc='- Validation')
    with torch.no_grad():
        for step, audio_data in enumerate(progress_bar):

            # Encode and decode audio_data
            codes = model.encode(audio_data)
            tgt_audio = model.decode(codes[..., 1:]).to(device)

            # Get src_codes and tgt_codes
            src_codes = codes[..., :-1].to(device)
            tgt_codes = codes[..., 1:].to(device)

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

            codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)  # shape: (B, n_codebooks, S, C)
            pred_codes = torch.argmax(codebook_index_probs, dim=-1)  # shape: (B, n_codebooks, S), just like input sequence!

            # Predict audio and codes
            pred_audio = model.decode(pred_codes)
            (pred_codes, pred_audio) = pred_codes.to(device), pred_audio.to(device)

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

    n_epochs = config['n_epochs']
    steps_per_epoch = config['steps_per_epoch']
    learning_rate = config['learning_rate']
    validation_metadata_path = config['validation_metadata_path']
    segment_dur = config['segment_dur']
    num_workers = config['num_workers']
    transformer_device = config["transformer"]['device']
    version = config['version']

    model = load_model(version, config)
    optimizer = torch.optim.Adam(model.transformer.parameters(), lr=learning_rate)

    version, last_epoch = resume_from_checkpoint(model, optimizer, version) if config["resume"] else ('1.0.0', 0)

    val_ds = ValidationDataset(sample_rate=model.sample_rate,
                               metadata_path=validation_metadata_path,
                               data_per_epoch=steps_per_epoch,  # batch_size * steps_per_epoch
                               segment_dur=segment_dur,
                               device=transformer_device)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    code_loss_fn = torch.nn.CrossEntropyLoss()
    audio_loss_fn = torch.nn.L1Loss()

    for epoch in range(last_epoch, n_epochs + last_epoch):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        avg_audio_loss, avg_code_loss = validation_loop(epoch, val_loader, model, audio_loss_fn, code_loss_fn, transformer_device)
        if config['save']: save(version, epoch + 1, model, optimizer, avg_audio_loss, avg_code_loss,
                                config['epoch_to_save'])

    print("Done!")
