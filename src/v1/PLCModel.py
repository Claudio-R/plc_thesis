import torch
from torch import Tensor
from copy import deepcopy
import dac
from .transformer import Transformer
from tqdm import tqdm

class PLCModel(torch.nn.Module):
    def __init__(self, config: dict):
        super(PLCModel, self).__init__()

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.sample_rate = config["sample_rate"]
        self.dac = dac.DAC.load(str(dac.utils.download(model_type="44khz"))).to(self.device)
        self.packet_dim = config["transformer"]["d_model"]

        self.transformer = Transformer(
            n_codebooks=self.dac.n_codebooks,
            codebook_size=self.dac.codebook_size,
            d_model=config["transformer"]['d_model'],
            d_attn=config["transformer"]['d_attn'],
            n_heads=config["transformer"]['n_heads'],
            n_layers=config["transformer"]['n_layers'],
            dropout=config["transformer"]['dropout'],
            dropout_attn=config["transformer"]['dropout_attn']
            ).to(self.device)

        # print('dac_is_cuda:', next(self.dac.parameters()).is_cuda)
        # print('transformer_is_cuda:', next(self.transformer.parameters()).is_cuda)

    @torch.no_grad()
    def code_transform(self, codes: torch.LongTensor) -> torch.Tensor:
        """ Maps codes from [0, 1023] (long) onto [0, 1] (float) """
        codes = codes / (self.dac.codebook_size - 1)
        return codes

    @torch.no_grad()
    def inverse_code_transform(self, codes: torch.FloatTensor) -> torch.Tensor:
        """ Maps codes from [0, 1] (float) onto [0, 1023] (long) """
        codes = codes * (self.dac.codebook_size - 1)
        codes = codes.round().long()
        return codes

    @torch.no_grad()
    def encode(self, audio_data: torch.Tensor) -> torch.Tensor:
        """ audio_data: (B, T) --> codes_long: (B, N, T) """
        audio_data = audio_data.to(self.device)
        x = self.dac.preprocess(audio_data, self.sample_rate)
        codes = self.dac.encode(x)[1]
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """ codes: (B, N, T) --> audio: (B, T) """
        # codes = codes.to(self.dac.device)
        z = self.dac.quantizer.from_codes(codes)[0]
        audio_data = self.dac.decode(z)
        return audio_data

    def forward(self, src_codes: torch.Tensor) -> torch.FloatTensor:
        """
            params:
            src_codes: FloatTensor(B, N, T) already transformed by the DataLoader

            returns:
            logits: Tensor(B, N, T, C)
        """
        src_codes = src_codes.to(self.device)
        logits = self.transformer(src_codes)

        return logits

    # @torch.no_grad()
    # def predict(self, src_codes: torch.FloatTensor) -> tuple[Tensor, Tensor]:
    #     """
    #     params
    #     - src_codes: FloatTensor[B, N, T] (already transformed by the DataLoader).
    #
    #     returns
    #     - predicted_audio: FloatTensor[B, T] (decoded by DAC)
    #     - predicted_codes: LongTensor[B, N, T] (decoded by DAC)
    #     """
    #     logist = self.forward(src_codes)
    #     pred_codes_long = self.transformer(pred_codes_float)
    #     pred_audio = self.decode(pred_codes_long)
    #     return pred_audio, pred_codes_float

    def inference(self, audio_data, audio_data_lost, trace) -> torch.Tensor:
        # t0: output_signal = deepcopy(audio_data_lost)
        codes_sequence = self.encode(audio_data)
        # t2: pred_codes_sequence = deepcopy(codes_sequence)

        for i, loss in enumerate(trace):
            if loss:
                # t0
                # idx = i * self.packet_dim
                # valid_sequence = output_signal[..., 0:idx]
                # codes_long = self.encode(valid_sequence)
                # codes_float = self.code_transform(codes_long).to(self.device)
                # pred_codes_float = self.forward(codes_float)
                # pred_packet_codes = pred_codes_float[..., -1].unsqueeze(-1)
                # pred_packet_audio = self.decode(self.inverse_code_transform(pred_packet_codes))
                # output_signal[..., idx:idx+self.packet_dim] = pred_packet_audio

                #t1 -
                src_codes = codes_sequence[..., :i]
                logits = self.forward(src_codes)
                codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_codes = torch.argmax(codebook_index_probs, dim=-1)
                codes_sequence[..., i] = pred_codes[..., -1]

                #t2 - Teacher Forcing
                # src_codes = codes_sequence[..., :i]
                # logits = self.forward(src_codes)
                # codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)  # shape: (B, n_codebooks, S, C)
                # pred_codes = torch.argmax(codebook_index_probs, dim=-1)  # shape: (B, n_codebooks, S), just like input sequence!
                # pred_codes_sequence[..., i] = pred_codes[..., -1]

        output_signal = self.decode(codes_sequence)
        return output_signal


