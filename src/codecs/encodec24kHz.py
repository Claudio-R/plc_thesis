import torch
import torchaudio as ta
import numpy as np
import encodec
from typing import Tuple

class EnCodec24kHz(torch.nn.Module):

    def __init__(self, kbps: float = 6., **kwargs):
        """
        The number of codebooks used will be determined by the bandwidth selected.
        E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
        Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
        For the 48 kHz model (non-causal), only 3, 6, 12, and 24 kbps are supported. The number
        of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
        :param kbps: bandwidth
        :param device: torch device
        """
        super().__init__(**kwargs)

        self.model = encodec.EncodecModel.encodec_model_24khz()

        assert kbps in self.model.target_bandwidths, f'Target bandwidth ({kbps} kbps) is not supported.'

        self.model.eval()
        self.kbps = kbps
        self.model.set_target_bandwidth(self.kbps)
        self.n_codebooks = int(2 ** np.ceil(np.log2(self.kbps)))
        self.codebook_size = self.model.quantizer.bins
        self.sample_rate = self.model.sample_rate
        self.frame_dim = 320

    def preprocess_audio(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # when installed with pip, encoded seem to have an assertion that prevents batch processing. Namely,
        # > assert wav.shape[0] in [1, 2] "Audio must be mono or stereo."
        # Here we reimplement "encodec.utils.convert_audio" with wav.shape[-2] instead of wav.shape[0]

        assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
        assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."

        *shape, channels, length = wav.shape

        target_channels = self.model.channels
        target_sr = self.model.sample_rate

        if target_channels == 1:
            wav = wav.mean(-2, keepdim=True)
        elif target_channels == 2:
            wav = wav.expand(*shape, target_channels, length)
        elif channels == 1:
            wav = wav.expand(target_channels, -1)
        else:
            raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")

        wav = ta.transforms.Resample(sr, target_sr)(wav)

        return wav

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded_frames = self.model.encode(audio)
            codes = encoded_frames[0][0]
        return codes

    def decode(self, codes: torch.Tensor) -> Tuple[torch.Tensor, int]:
        with torch.no_grad():
            encoded_frames = [(codes, None)]
            audio = self.model.decode(encoded_frames)
        return audio