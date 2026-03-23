from pytorch_lightning.callbacks import Callback


class CurriculumChannelSamplingCallback(Callback):
    def __init__(self, n_channels: int):
        self.n_channels = n_channels

    def on_train_epoch_start(self, trainer, pl_module):
        if not hasattr(pl_module, 'encoder'):
            return
        encoder = pl_module.encoder
        if not hasattr(encoder, 'patch_embed') or not hasattr(encoder.patch_embed, 'min_sample_channels'):
            return
        epoch_progress = trainer.current_epoch / max(1, trainer.max_epochs - 1)
        min_ch = max(1, int(self.n_channels * epoch_progress * 0.5))
        encoder.patch_embed.min_sample_channels = min_ch
