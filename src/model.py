import os, yaml
from copy import deepcopy
import lightning as L
import torchmetrics
import soundfile as sf
import src.utils as utils
from src.codecs.encodec24kHz import EnCodec24kHz
from src.utils import load_transformer, get_mode
import torch
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

class Model(L.LightningModule):
    def __init__(self):
        # Model
        super().__init__()
        with open('config.yaml') as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)

        self.model_mode = config['model_mode']
        self.codec = EnCodec24kHz(config['codec']['kbps'])
        self.codebook_size = self.codec.codebook_size if self.model_mode == 'parallel' else self.codec.codebook_size+1
        self.version = config['version']
        self.transformer = load_transformer(config,
                                       self.codec.n_codebooks,
                                       self.codebook_size,
                                       self.codec.sample_rate,
                                       self.codec.frame_dim)
        # self.model_mode = 'parallel'
        self.mode = get_mode(self.version)
        self.nq = self.codec.n_codebooks
        self.pad_length = self.nq - 1
        self.special_token = 1024
        # meaningful only in range [nq, 1]: nq: predict nq steps forward in a step, 1, predict just one step forward
        self.forward_steps = self.nq - 0

        # Loss functions
        self.code_loss_fn = torch.nn.CrossEntropyLoss()
        self.audio_loss_fn = torch.nn.L1Loss()

        # Metrics
        self.accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=self.codebook_size)
        self.stoi_fn = ShortTimeObjectiveIntelligibility(self.codec.sample_rate, False)

        # Folders
        self.test_dir = f'test/plc_challenge'
        self.clean_dir = f'{self.test_dir}/clean'
        self.lossy_dir = f'{self.test_dir}/lossy'
        self.enhanced_dir = f'{self.test_dir}/enhanced/{self.model_mode}/{self.mode}/{self.transformer.version}'
        self.traces_dir = f'{self.test_dir}/traces'
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.lossy_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)
        os.makedirs(self.traces_dir, exist_ok=True)

    def configure_optimizers(self):
        for param in self.codec.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer

    def forward(self, codes, trace):
        # Packet loss
        pred_codes = deepcopy(codes)
        codes_lost_only = []
        pred_codes_lost_only = []

        for i, loss in enumerate(trace):
            if loss:
                first_idx = max(0, i - self.transformer.context_length)
                src_codes = codes[..., first_idx:i]
                logits = self.transformer(src_codes)
                codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_lost_codes = torch.argmax(codebook_index_probs, dim=-1)
                if self.model_mode == 'parallel':
                    pred_codes[..., i] = pred_lost_codes[..., -1]
                elif self.model_mode == 'delayed':
                    for j in range(self.nq):
                        pred_codes[..., j, i] = pred_lost_codes[..., j, -self.nq+j]
                else:
                    raise NotImplementedError
                codes_lost_only.append(codes[..., i])
                pred_codes_lost_only.append(pred_codes[..., i])

        # Decoding
        pred_audio = self.codec.decode(pred_codes)

        codes_lost_only = torch.stack(codes_lost_only)
        pred_codes_lost_only = torch.stack(pred_codes_lost_only)
        return pred_audio, pred_codes, (codes_lost_only, pred_codes_lost_only)

    def training_step(self, batch, batch_idx):
        self.codec.eval()
        codes = self.codec.encode(batch)
        src_codes, tgt_codes = self.split_codes(codes)
        logits = self.transformer(src_codes)
        code_loss = self.compute_code_loss(logits, tgt_codes)
        self.log(name="code_loss", value=code_loss, prog_bar=True)
        return code_loss

    def validation_step(self, batch, batch_idx):
        codes = self.codec.encode(batch)
        src_codes, tgt_codes = self.split_codes(codes)
        logits, pred_codes = self.transformer.predict(src_codes)
        code_loss = self.compute_code_loss(logits, tgt_codes)
        if self.model_mode == 'delayed':
            tgt_codes = self.rollback_codes(tgt_codes)
            pred_codes = self.rollback_codes(pred_codes)

        tgt_audio = self.codec.decode(tgt_codes)
        pred_audio = self.codec.decode(pred_codes)
        audio_loss = self.compute_audio_loss(pred_audio, tgt_audio)
        logs = {
            'code_loss': code_loss,
            'audio_loss': audio_loss,
            'accuracy': self.accuracy_fn(pred_codes, tgt_codes),
            'stoi': self.stoi_fn(pred_audio, tgt_audio)
        }
        self.log_dict(logs)
        return audio_loss

    def test_step(self, batch, batch_idx):
        codes = self.codec.encode(batch)
        src_codes, tgt_codes = self.split_codes(codes)
        logits, pred_codes = self.transformer.predict(src_codes)
        code_loss = self.compute_code_loss(logits, tgt_codes)
        if self.model_mode == 'delayed':
            tgt_codes = self.rollback_codes(tgt_codes)
            pred_codes = self.rollback_codes(pred_codes)

        tgt_audio = self.codec.decode(tgt_codes)
        pred_audio = self.codec.decode(pred_codes)
        audio_loss = self.compute_audio_loss(pred_audio, tgt_audio)
        logs = {
            'code_loss': code_loss,
            'audio_loss': audio_loss,
            'accuracy': self.accuracy_fn(pred_codes, tgt_codes),
            'stoi': self.stoi_fn(pred_audio, tgt_audio)
        }
        self.log_dict(logs)
        return logs

    def predict_step(self, batch, batch_idx):
        # batch: audio+trace
        audio, trace = batch
        trace = trace.squeeze()

        # Encoding
        codes = self.codec.encode(audio)
        tgt_audio = self.codec.decode(codes)

        # Packet loss
        codes_lost = utils.simulate_packet_loss(codes, trace)
        tgt_audio_lost = self.codec.decode(codes_lost)

        # Predict
        pred_audio, pred_codes, (y, z) = self.forward(codes, trace)

        # Losses and metrics
        logs = {
            'audio_loss': self.compute_audio_loss(pred_audio, tgt_audio),
            'prediction accuracy': self.accuracy_fn(z, y),
            'stoi': self.stoi_fn(pred_audio, tgt_audio)
        }
        # self.log_dict(logs)

        # Save audio files and traces
        sr = self.codec.sample_rate
        sf.write(f'{self.clean_dir}/tgt_audio_{batch_idx}.wav', tgt_audio.squeeze().to('cpu'), sr)
        sf.write(f'{self.lossy_dir}/tgt_audio_{batch_idx}.wav', tgt_audio_lost.squeeze().to('cpu'), sr)
        sf.write(f'{self.enhanced_dir}/tgt_audio_{batch_idx}.wav', pred_audio.squeeze().to('cpu'), sr)
        with open(f'{self.traces_dir}/tgt_audio_{batch_idx}.txt', 'w') as f:
            for trace_idx in trace:
                f.write('{}\n'.format(trace_idx))
        return logs

    def split_codes(self, codes):
        if self.model_mode == 'parallel':
            src_codes = codes[..., :-1]
            tgt_codes = codes[..., 1:]
        elif self.model_mode == 'delayed':
            padding_tensor = torch.ones((codes.shape[0], codes.shape[1], self.pad_length)).type_as(codes) * self.special_token
            codes = torch.cat((codes, padding_tensor), dim=-1)
            for i in range(1, self.nq):
                codes[:, i:, :] = torch.roll(codes[:, i:, :], shifts=1, dims=-1)
            src_codes = codes[..., :-self.nq]
            tgt_codes = codes[..., self.forward_steps:self.forward_steps+src_codes.shape[-1]] # guarantees shape is the same
        else:
            raise NotImplementedError
        return src_codes, tgt_codes

    def rollback_codes(self, delayed_codes):
        for i in range(1, self.nq):
            delayed_codes[:, i:, :] = torch.roll(delayed_codes[:, i:, :], shifts=-1, dims=-1)
        delayed_codes = delayed_codes[..., :-(self.nq - 1)]
        # for pred_codes, remove residual special tokensf
        delayed_codes[delayed_codes == self.special_token] = 0
        return delayed_codes

    def compute_code_loss(self, logits, tgt_codes):
        code_loss = sum([self.code_loss_fn(
            logits[:, k, :, :].contiguous().view(-1, logits.size(-1)),
            tgt_codes[:, k, :].contiguous().view(-1))
            for k in range(self.codec.n_codebooks)])
        code_loss /= self.codec.n_codebooks
        return code_loss

    def compute_audio_loss(self, pred_audio, tgt_audio):
        return self.audio_loss_fn(pred_audio, tgt_audio)