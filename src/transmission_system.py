import torch
from typing import Protocol

class TransmissionSystem(torch.nn.Module):
    def __init__(self, codec, transformer, sample_rate:int = 44100, packet_size:int = 512) -> None:
        super(TransmissionSystem, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.codec = codec.to(self.device)
        self.transformer = transformer.to(self.device)
        self.sample_rate = sample_rate
        self.packet_size = packet_size

    @torch.no_grad()
    def encode(self, audio_data: torch.Tensor):
        """
        Encodes audio packets to a batch of codes
        :param audio_data: audio packets (B, T)
        :return: codes (B, N, S)
        """
        
        codes = self.codec.encode(audio_data)
        src_codes = codes[..., :-1]
        tgt_codes = codes[..., 1:]
        return src_codes, tgt_codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decodes codes to audio packets
        :param codes: codes to decode (B, N, S)
        :return: decoded codes (B, T)
        """
        pass

    def forward(self, audio_data: torch.Tensor):
        """
        Performs forward pass of the transformer model
        params: audio_data: batch of audio packets
        returns: pred_codes: predicted codes
        """
        codes = self.encoder(audio_data)
        pred_codes, code_loss = self.transformer(codes)
        return pred_codes, code_loss

    @torch.no_grad()
    def transmit(self, tgt_audio_data, trace) -> torch.Tensor:
        """
        Performs prediction on the whole lossy audio
        :param tgt_audio_data: full target audio data
        :param trace: loss_indicator
        :return: predicted audio data
        """

        codes_sequence = self.encode(tgt_audio_data)

        for i, is_lost in enumerate(trace):
            if is_lost:
                src_codes = codes_sequence[..., :i]
                pred_codes = self.transformer(src_codes)
                codes_sequence[..., i] = pred_codes[..., -1]

        pred_audio_data = self.decode(codes_sequence)
        return pred_audio_data


