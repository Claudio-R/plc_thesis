import os
import torch
import lightning as L
import torchmetrics
import soundfile as sf
import src.utils as utils

# define the LightningModule
class Model(L.LightningModule):
    def __init__(self, config):
        # Model
        super().__init__()
        self.codec = utils.load_codec()
        self.transformer = utils.load_transformer(config)
        self.mode = 'parallel' # ['parallel', 'delayed']

        # Loss functions
        self.code_loss_fn = torch.nn.CrossEntropyLoss()
        self.audio_loss_fn = torch.nn.L1Loss()

        # Metrics
        self.accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=self.codec.codebook_size)
        self.pesq_fn = torchmetrics.audio.PerceptualEvaluationSpeechQuality(self.codec.sample_rate, 'wb')
        self.stoi_fn = torchmetrics.audio.ShortTimeObjectiveIntelligibility(self.codec.sample_rate, False)

        # Folders
        version_id = self.transformer.version.split('.')[0]
        self.test_dir = f'test/{self.mode}'
        self.clean_dir = f'{self.test_dir}/clean'
        self.lossy_dir = f'{self.test_dir}/lossy'
        self.enhanced_dir = f'{self.test_dir}/enhanced/model_v{version_id}/{self.transformer.version}'
        self.traces_dir = f'{self.test_dir}/traces'
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.lossy_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)
        os.makedirs(self.traces_dir, exist_ok=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def forward(self, audio, audio_idx, trace):
        # Encoding
        codes = self.codec.encode(audio)

        # Packet loss
        codes_lost = utils.simulate_packet_loss(codes, trace, packet_dim=self.codec.frame_dim)
        tgt_audio_lost = self.codec.decode(codes_lost)

        # Inference
        for i, loss in enumerate(trace[0, :]):
            if loss:
                if self.mode == 'parallel':
                    first_idx = max(0, i - self.transformer.context_length)
                    src_codes = codes[..., first_idx:i]
                    logits = self.transformer(src_codes)
                    codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)
                    pred_codes = torch.argmax(codebook_index_probs, dim=-1)
                    codes_lost[..., i] = pred_codes[..., -1]
                else:
                    # TODO: delayed inference
                    raise Exception('not implemented')

        # Decoding
        pred_audio = self.codec.decode(codes_lost)
        tgt_audio = self.codec.decode(codes)

        # Losses and metrics
        audio_loss = self.compute_audio_loss(pred_audio, tgt_audio)
        logs = {
            'audio_loss': audio_loss,
            'pesq': self.pesq_fn(pred_audio, tgt_audio),
            'stoi': self.stoi_fn(pred_audio, tgt_audio)
        }
        self.log_dict(logs)

        # Save audio files and traces
        sr = self.codec.sample_rate
        sf.write(f'{self.clean_dir}/tgt_audio_{audio_idx}.wav', tgt_audio.squeeze().to('cpu'), sr)
        sf.write(f'{self.lossy_dir}/tgt_audio_{audio_idx}.wav', tgt_audio_lost.squeeze().to('cpu'), sr)
        sf.write(f'{self.enhanced_dir}/tgt_audio_{audio_idx}.wav', pred_audio.squeeze().to('cpu'), sr)
        with open(f'{self.traces_dir}/tgt_audio_{audio_idx}.txt', 'w') as f:
            for trace_idx in trace:
                f.write('{}\n'.format(trace_idx))
        return logs


    def training_step(self, batch, batch_idx):
        codes = self.codec.encode(batch)  # (64, 8, 150)
        src_codes, tgt_codes = self.split_codes(codes)
        logits = self.transformer(src_codes)
        code_loss = self.compute_code_loss(logits, tgt_codes)
        self.log(name="code_loss", value=code_loss, prog_bar=True)
        return code_loss


    def validation_step(self, batch, batch_idx):
        codes = self.codec.encode(batch)
        src_codes, tgt_codes = self.split_codes(codes)
        logits, pred_codes = self.transformer.predict(src_codes)
        tgt_audio = self.codec.decode(codes)
        pred_audio = self.codec.decode(pred_codes)
        code_loss = self.compute_code_loss(logits, tgt_codes)
        audio_loss = self.compute_audio_loss(pred_audio, tgt_audio)
        logs = {
            'code_loss': code_loss,
            'audio_loss': audio_loss,
            'accuracy': self.accuracy_fn(pred_codes, tgt_codes),
            'pesq': self.pesq_fn(pred_audio, tgt_audio),
            'stoi': self.stoi_fn(pred_audio, tgt_audio)
        }
        self.log_dict(logs)
        return logs


    def split_codes(self, codes):
        if self.parallel:
            src_codes = codes[..., :-1]
            tgt_codes = codes[..., 1:]
        else:
            nq = self.transformer.n_codebooks
            src_codes = codes[..., :-nq]
            for i in range(1, nq):
                codes[:, i:, :] = torch.roll(codes[:, i:, :], shifts=1, dims=-1)
            tgt_codes = codes[..., nq:, :]
        return src_codes, tgt_codes


    def compute_code_loss(self, logits, tgt_codes):
        code_loss = torch.mean(
            torch.Tensor([
                self.code_loss_fn(
                    logits[:,k,:,:].view(-1, logits.size(-1)),
                    tgt_codes[:,k,:,:].view(-1, tgt_codes.size(-1)))
                for k in range(self.transformer.n_codebooks)
            ]))
        return code_loss


    def compute_audio_loss(self, pred_audio, tgt_audio):
        return self.audio_loss_fn(pred_audio, tgt_audio)