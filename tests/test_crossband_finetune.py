import random
import types
from unittest.mock import patch

import torch
import torch.nn as nn

from callbacks import CurriculumChannelSamplingCallback
from change_detection_pytorch.encoders.chi_vit import chivit_tiny
from classifier_utils import (
    ChannelDropout,
    adapt_encoder_for_multiband_eval,
    adapt_rgb_conv3d_layer_to_multiband_preserve_rgb,
    adapt_rgb_conv_layer_to_multiband,
    adapt_rgb_conv_layer_to_multiband_preserve_rgb,
    register_channel_embed_gradient_mask,
)
from utils import get_spectral_init_weights


def test_channel_dropout_eval_mode_passthrough():
    cd = ChannelDropout(p=0.5, min_channels=1)
    cd.eval()
    x = torch.randn(2, 4, 8, 8)
    out = cd(x)
    assert torch.equal(out, x)


def test_channel_dropout_p_zero_passthrough():
    cd = ChannelDropout(p=0.0, min_channels=1)
    cd.train()
    x = torch.randn(2, 4, 8, 8)
    out = cd(x)
    assert torch.equal(out, x)


def test_channel_dropout_train_reduces_channels():
    cd = ChannelDropout(p=0.5, min_channels=1)
    cd.train()
    B, C, H, W = 4, 6, 4, 4
    keep = max(1, int(C * (1 - 0.5)))
    torch.manual_seed(42)
    x = torch.ones(B, C, H, W)
    out = cd(x)
    for b in range(B):
        nonzero = (out[b].abs().sum(dim=(1, 2)) > 1e-6).sum().item()
        assert nonzero == keep
    scale = C / keep
    for b in range(B):
        kept = out[b].abs().sum(dim=(1, 2)) > 1e-6
        assert torch.allclose(out[b][kept].mean(), torch.tensor(scale, dtype=out.dtype), atol=1e-5)


def test_channel_dropout_min_channels_respected():
    cd = ChannelDropout(p=0.9, min_channels=2)
    cd.train()
    B, C, H, W = 2, 4, 4, 4
    keep = max(2, int(C * 0.1))
    torch.manual_seed(123)
    x = torch.ones(B, C, H, W)
    out = cd(x)
    for b in range(B):
        nonzero = (out[b].abs().sum(dim=(1, 2)) > 1e-6).sum().item()
        assert nonzero == keep


def test_get_spectral_init_weights_b08():
    w = get_spectral_init_weights("B08", ["B04", "B03", "B02"])
    assert abs(w[0] - 0.6) < 1e-6
    assert abs(w[1] - 0.3) < 1e-6
    assert abs(w[2] - 0.1) < 1e-6
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_get_spectral_init_weights_vv_uniform():
    w = get_spectral_init_weights("VV", ["B04", "B03", "B02"])
    assert len(w) == 3
    for v in w.values():
        assert abs(v - 1.0 / 3) < 1e-6
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_adapt_conv2d_preserve_rgb_mean_init():
    old = nn.Conv2d(3, 8, kernel_size=1)
    torch.manual_seed(1)
    nn.init.xavier_uniform_(old.weight)
    new = adapt_rgb_conv_layer_to_multiband_preserve_rgb(old, new_in_channels=4)
    assert new.in_channels == 4
    assert torch.allclose(new.weight[:, :3], old.weight)
    expected_mean = old.weight.mean(dim=1, keepdim=True).expand(-1, 1, 1, 1)
    assert torch.allclose(new.weight[:, 3:4], expected_mean)


def test_adapt_conv2d_preserve_rgb_spectral_init():
    old = nn.Conv2d(3, 8, kernel_size=1)
    torch.manual_seed(2)
    nn.init.xavier_uniform_(old.weight)
    new = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
        old,
        new_in_channels=4,
        spectral_init=True,
        training_bands=["B04", "B03", "B02"],
        new_bands=["B08"],
    )
    w = get_spectral_init_weights("B08", ["B04", "B03", "B02"])
    expected = sum(old.weight[:, j] * wt for j, wt in w.items())
    assert torch.allclose(new.weight[:, 3], expected)


def test_adapt_conv3d_preserve_rgb_mean_init():
    old = nn.Conv3d(3, 4, kernel_size=(1, 1, 1))
    torch.manual_seed(3)
    nn.init.xavier_uniform_(old.weight)
    new = adapt_rgb_conv3d_layer_to_multiband_preserve_rgb(old, new_in_channels=4)
    assert new.in_channels == 4
    assert torch.allclose(new.weight[:, :3], old.weight)
    expected_mean = old.weight.mean(dim=1, keepdim=True).expand(-1, 1, 1, 1, 1)
    assert torch.allclose(new.weight[:, 3:4], expected_mean)


def test_adapt_conv3d_preserve_rgb_spectral_init():
    old = nn.Conv3d(3, 4, kernel_size=(1, 1, 1))
    torch.manual_seed(4)
    nn.init.xavier_uniform_(old.weight)
    new = adapt_rgb_conv3d_layer_to_multiband_preserve_rgb(
        old,
        new_in_channels=4,
        spectral_init=True,
        training_bands=["B04", "B03", "B02"],
        new_bands=["B08"],
    )
    w = get_spectral_init_weights("B08", ["B04", "B03", "B02"])
    expected = sum(old.weight[:, j] * wt for j, wt in w.items())
    assert torch.allclose(new.weight[:, 3], expected)


def test_adapt_conv2d_to_multiband_expand_fills_mean():
    old = nn.Conv2d(3, 8, kernel_size=1)
    torch.manual_seed(11)
    nn.init.xavier_uniform_(old.weight)
    new = adapt_rgb_conv_layer_to_multiband(old, new_in_channels=6)
    assert new.in_channels == 6
    assert torch.allclose(new.weight[:, :3], old.weight)
    init_w = old.weight.mean(dim=1)
    for i in range(3, 6):
        assert torch.allclose(new.weight[:, i], init_w)


def test_adapt_conv2d_to_multiband_shrink_slices():
    old = nn.Conv2d(5, 4, kernel_size=1)
    torch.manual_seed(12)
    nn.init.xavier_uniform_(old.weight)
    new = adapt_rgb_conv_layer_to_multiband(old, new_in_channels=2)
    assert new.in_channels == 2
    assert torch.allclose(new.weight, old.weight[:, :2])


@patch("builtins.print")
def test_adapt_encoder_for_multiband_eval_vit_conv2d(mock_print):
    encoder = types.SimpleNamespace()
    encoder.patch_embed = types.SimpleNamespace()
    old = nn.Conv2d(3, 8, kernel_size=1)
    nn.init.xavier_uniform_(old.weight)
    encoder.patch_embed.proj = old
    encoder.patch_embed.in_chans = 3
    encoder.in_chans = 3
    adapt_encoder_for_multiband_eval(encoder, multiband_channel_count=4)
    assert encoder.patch_embed.proj.in_channels == 4
    assert encoder.patch_embed.in_chans == 4
    assert encoder.in_chans == 4
    assert torch.allclose(encoder.patch_embed.proj.weight[:, :3], old.weight)


@patch("builtins.print")
def test_adapt_encoder_for_multiband_eval_vit_conv3d(mock_print):
    encoder = types.SimpleNamespace()
    encoder.patch_embed = types.SimpleNamespace()
    old = nn.Conv3d(3, 8, kernel_size=(1, 1, 1))
    nn.init.xavier_uniform_(old.weight)
    encoder.patch_embed.proj = old
    encoder.patch_embed.in_chans = 3
    adapt_encoder_for_multiband_eval(encoder, multiband_channel_count=4)
    assert encoder.patch_embed.proj.in_channels == 4
    assert encoder.patch_embed.in_chans == 4
    assert torch.allclose(encoder.patch_embed.proj.weight[:, :3], old.weight)


@patch("builtins.print")
def test_chivit_pooling_mode_cls(mock_print):
    torch.manual_seed(42)
    model = chivit_tiny(
        in_chans=3,
        img_size=[224],
        patch_size=16,
        add_ch_embed=True,
        enable_sample=False,
        pooling_mode="cls",
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    channel_idxs = [0, 1, 2]
    out = model(x, channel_idxs)
    assert out.shape == (2, 192)


@patch("builtins.print")
def test_chivit_pooling_mode_channel_mean(mock_print):
    torch.manual_seed(43)
    model = chivit_tiny(
        in_chans=3,
        img_size=[224],
        patch_size=16,
        add_ch_embed=True,
        enable_sample=False,
        pooling_mode="channel_mean",
    )
    model.eval()
    x3 = torch.randn(2, 3, 224, 224)
    x2 = x3[:, :2].clone()
    out3 = model(x3, [0, 1, 2])
    out2 = model(x2, [0, 1])
    assert out3.shape == (2, 192)
    assert out2.shape == (2, 192)
    assert not torch.allclose(out3, out2, atol=1e-3)


@patch("builtins.print")
def test_chivit_hcs_channel_idxs_remapping(mock_print):
    torch.manual_seed(0)
    model = chivit_tiny(
        in_chans=12,
        img_size=[32],
        patch_size=16,
        add_ch_embed=True,
        enable_sample=True,
        min_sample_channels=1,
    )
    model.train()
    x = torch.randn(1, 3, 32, 32)
    band_idxs = [2, 5, 8]

    def fake_randint(a: int, b: int) -> int:
        return 1

    def fake_sample(population, k: int) -> list[int]:
        return [1]

    with patch("change_detection_pytorch.encoders.chi_vit.random.randint", side_effect=fake_randint), patch(
        "change_detection_pytorch.encoders.chi_vit.random.sample", side_effect=fake_sample
    ):
        ce = model.patch_embed.channel_embed
        with torch.no_grad():
            ce.zero_()
            ce[:, :, 5, :, :] = 100.0
        out_a = model.patch_embed(x.clone(), band_idxs)
        with torch.no_grad():
            ce.zero_()
            ce[:, :, 1, :, :] = 100.0
        out_b = model.patch_embed(x.clone(), band_idxs)
    assert out_a.mean() > out_b.mean() + 10.0


@patch("builtins.print")
def test_chivit_channel_gate_kills_channel(mock_print):
    torch.manual_seed(42)
    model = chivit_tiny(
        in_chans=3,
        img_size=[224],
        patch_size=16,
        add_ch_embed=True,
        enable_channel_gate=True,
    )
    model.patch_embed.channel_gate.data[0] = -100.0
    x = torch.randn(2, 3, 224, 224)
    x2 = x.clone()
    x2[:, 0] = 0.0
    channel_idxs = [0, 1, 2]
    out1 = model(x, channel_idxs)
    out2 = model(x2, channel_idxs)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_curriculum_callback_updates_min_sample_channels():
    pl_module = types.SimpleNamespace()
    pl_module.encoder = types.SimpleNamespace()
    pl_module.encoder.patch_embed = types.SimpleNamespace()
    pl_module.encoder.patch_embed.min_sample_channels = 1
    trainer = types.SimpleNamespace(current_epoch=0, max_epochs=10)
    cb = CurriculumChannelSamplingCallback(n_channels=6)
    cb.on_train_epoch_start(trainer, pl_module)
    assert pl_module.encoder.patch_embed.min_sample_channels == 1
    trainer.current_epoch = 9
    cb.on_train_epoch_start(trainer, pl_module)
    expected = max(1, int(6 * (9 / 9) * 0.5))
    assert pl_module.encoder.patch_embed.min_sample_channels == expected


def test_register_channel_embed_gradient_mask_zeroes_non_training_grads():
    embed = nn.Parameter(torch.randn(1, 12, 8, 1, 1))
    register_channel_embed_gradient_mask(embed, training_channel_idxs={0, 1, 2})
    loss = embed.sum()
    loss.backward()
    assert embed.grad is not None
    for idx in range(8):
        if idx in {0, 1, 2}:
            assert embed.grad[0, 0, idx, 0, 0].abs().sum() > 0
        else:
            assert embed.grad[0, 0, idx, 0, 0].abs().sum() == 0
