from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class LogMelSpectrogram(nn.Module):
    """
    Create spectrogram from raw audio and make
    that logarithmic for avoiding inf values.
    """
    def __init__(self, sample_rate=16000, n_mels=40):
        super(LogMelSpectrogram, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, 
                                                              n_fft=1024, hop_length=256, f_min=0, f_max=8000)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.transform(waveform)
        log_mel = torch.log(spectrogram + 1e-9)
        return (log_mel - log_mel.mean(dim=1, keepdim=True)) / \
            (log_mel.std(dim=1,keepdim=True) + 1e-9)


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def upscale_to_wav_len(mask):
    hop_len = 256
    return mask.repeat_interleave(hop_len)
