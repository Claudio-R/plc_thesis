import os
import numpy as np
from copy import deepcopy
import librosa
import torch
import pandas as pd
import random
import math
import src.v1
import src.v2
from .encodec24kHz import EnCodec24kHz

def resume_from_checkpoint(model, optimizer, current_version:str, map_location:str='cuda:1'):

    version_id = current_version.split('.')[0]

    ckp = dict()
    version_dir = f'meta/checkpoint/v{version_id}'
    os.makedirs(version_dir, exist_ok=True)
    v_folders = os.listdir(version_dir)

    if len(v_folders) == 0:
        print(f'No available checkpoints')
        return current_version, 0

    print('available_checkpoints:')
    for v in v_folders:
        model_ckp = [f for f in os.listdir(f'{version_dir}/{v}') if f.startswith('model_')]
        epochs = [int(ckp_file.split('.')[0].split('_e')[1]) for ckp_file in model_ckp]
        print(f'- {v}: {epochs}')
        ckp[v] = epochs

    version = input("Enter version to resume: ")
    if 'v'+version not in ckp.keys():
        print(Exception(f'Invalid version. Exiting.'))
        return current_version, 0

    epoch = int(input("Enter epoch to resume: "))
    if epoch not in ckp['v'+version]:
        latest_epoch = max(ckp['v' + version])
        print(Exception(f'Invalid epoch. Loading from latest epoch: e{latest_epoch}'))
        return version, latest_epoch

    print(f'Resuming model at version {version} and epoch {epoch}')

    version_dir = f'{version_dir}/v{version}'
    model_ckp = os.path.join(version_dir, 'model_e' + str(epoch) + '.pth')
    optimizer_ckp = os.path.join(version_dir,'optimizer_e' + str(epoch) + '.pth')

    try:
        if model: model.load_state_dict(torch.load(model_ckp, map_location=map_location))
        if optimizer: optimizer.load_state_dict(torch.load(optimizer_ckp, map_location=map_location))
    except Exception as e:
        print(e)
        print(f'Could not load model from {version_dir} at epoch {epoch}. Exiting.')
        return current_version, 0

    return version, epoch

def load_codec(codec_name:str='encodec', config=None):
    if codec_name == 'encodec':
        return EnCodec24kHz()
    else:
        raise ValueError(f'Please provide a valid codec: {codec_name}')

def load_transformer(version, config):
    version_id = version.split('.')[0]
    if version_id == '1':
        return src.v1.Transformer()
    else:
        return src.v2.Transformer(config)
def save(version, epoch, model, optimizer, avg_audio_loss, avg_code_loss, epoch_to_save):

    version_id = version.split('.')[0]

    # Save checkpoints
    if epoch % epoch_to_save == 0:
        pth_dir = f'meta/checkpoint/v{version_id}/v{version}'
        os.makedirs(pth_dir, exist_ok=True)
        pth_model = os.path.join(pth_dir, f'model_e{epoch}.pth')
        pth_optimizer = os.path.join(pth_dir, f'optimizer_e{epoch}.pth')
        torch.save(model.state_dict(), pth_model)
        torch.save(optimizer.state_dict(), pth_optimizer)

    pth_dir = f'meta/history/v{version_id}/v{version}'
    os.makedirs(pth_dir, exist_ok=True)

    # Save avg_loss
    pth_loss = os.path.join(pth_dir, 'loss_history.csv')
    try:
        hdl = pd.read_csv(pth_loss)
    except:
        hdl = pd.DataFrame({'version': [], 'epoch': [], 'avg_audio_loss': [], 'avg_code_loss': []})
    hdl.loc[len(hdl.index)] = [version, epoch, avg_audio_loss, avg_code_loss]
    hdl.to_csv(pth_loss, index=False)

    # Save best_audio_loss
    pth_bestLoss = os.path.join(pth_dir, 'best_audio_loss_history.csv')
    try:
        hdl = pd.read_csv(pth_bestLoss)
        curr_best_loss = min(hdl['best_audio_loss'])
    except:
        hdl = pd.DataFrame({'version': [], 'epoch': [], 'best_audio_loss': []})
        curr_best_loss = float('inf')
    if avg_audio_loss < float(curr_best_loss):
        best_audio_loss = avg_audio_loss
        print("Current best_audio_loss: {0:.6g}".format(curr_best_loss))
        print("New best_audio_loss: {0:.6g}".format(best_audio_loss))
        hdl.loc[len(hdl.index)] = [version, epoch, best_audio_loss]
    hdl.to_csv(pth_bestLoss, index=False)


def create_trace(audio_test_path, packet_dim:int=512, sr:int=44100, loss_rate: int=10, random_trace:bool = False):
    """
    Create a trace
    :param audio_test_path:
    :param packet_dim:
    :param sr:
    :param loss_rate:
    :param random_trace: whether to randomly sample the loss
    :return: Numpy array
    """

    # load the clean signal
    y_true, sr = librosa.load(audio_test_path, sr=sr, mono=True)

    # ----------- Simulate packet losses ----------- #
    # Define the trace of lost packets: 1s indicate a loss
    trace_len = math.ceil(len(y_true) // packet_dim)
    trace = np.zeros(trace_len, dtype=int)
    if not random_trace:
        trace[np.arange(loss_rate, trace_len, loss_rate)] = 1
    else:
        for idx in range(1, trace_len):
            prob = 0.3
            trace[idx] = 0 if random.uniform(0, 1) > prob else 1
    return trace


def simulate_packet_loss(y_ref: np.ndarray, trace: np.ndarray, packet_dim: int=512) -> np.ndarray:
    # Copy the clean signal to create the lossy signal
    y_lost = deepcopy(y_ref)

    # Simulate packet losses according to given trace
    for i, loss in enumerate(trace):
        if loss:
            idx = i * packet_dim
            y_lost[..., idx: idx + packet_dim] = 0.

    return y_lost
