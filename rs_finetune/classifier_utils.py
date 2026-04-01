import timm
import torch
import torch.nn as nn

from change_detection_pytorch.encoders import (
    anysat_encoders,
    chi_vit_encoders,
    clay_encoders,
    croma_encoders,
    dinov2_encoders,
    dofa_encoders,
    prithvi_encoders,
    swin_transformer_encoders,
    terrafm_encoders,
    timm_resnet_encoders,
    timm_vit_encoders,
    vit_encoders,
)
from change_detection_pytorch.encoders._utils import load_pretrained
from change_detection_pytorch.encoders.dinov3 import SharedChannelPatchConv
from utils import get_spectral_init_weights

timm_encoders = timm_vit_encoders.copy()
timm_encoders.update(timm_resnet_encoders)


def register_channel_embed_gradient_mask(embed_param: nn.Parameter, training_channel_idxs: set[int]) -> None:
    mask = torch.zeros(embed_param.shape[2], dtype=torch.float32, device=embed_param.device)
    for idx in training_channel_idxs:
        mask[idx] = 1.0
    embed_param.register_hook(lambda g, m=mask: g * m.view(1, 1, -1, 1, 1).to(g.device))


class ChannelDropout(nn.Module):
    def __init__(self, p: float = 0.2, min_channels: int = 1):
        super().__init__()
        self.p = p
        self.min_channels = min_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        B, C, H, W = x.shape
        keep = max(self.min_channels, int(C * (1 - self.p)))
        mask = torch.zeros(B, C, 1, 1, device=x.device, dtype=x.dtype)
        for b in range(B):
            idx = torch.randperm(C, device=x.device)[:keep]
            mask[b, idx] = C / keep
        return x * mask


def adapt_rgb_conv_layer_to_multiband(old_conv: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:
    old_in_channels = old_conv.in_channels
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    old_weights = old_conv.weight.data
    new_weights = torch.zeros(
        old_conv.out_channels,
        new_in_channels,
        *old_conv.kernel_size,
        device=old_weights.device,
        dtype=old_weights.dtype,
    )
    if new_in_channels >= old_in_channels:
        new_weights[:, :old_in_channels, :, :] = old_weights
        if new_in_channels > old_in_channels:
            init_w = old_weights.mean(dim=1)
            for i in range(old_in_channels, new_in_channels):
                new_weights[:, i, :, :] = init_w
    else:
        new_weights = old_weights[:, :new_in_channels, :, :].clone()
    new_conv.weight.data.copy_(new_weights)

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)

    return new_conv


def adapt_rgb_conv_layer_to_multiband_preserve_rgb(
    old_conv: nn.Conv2d,
    new_in_channels: int = 4,
    spectral_init: bool = False,
    training_bands: list[str] | None = None,
    new_bands: list[str] | None = None,
) -> nn.Conv2d:
    """
    Adapt a 3-channel RGB conv layer to a new number of channels, preserving the original weights.
    For each new channel, initialize its weights as a weighted average of the training bands.

    Args:
        old_conv: Original convolution layer
        new_in_channels: New number of input channels
        spectral_init: whether to use spectral initialization
        training_bands: the training bands
        new_bands: the new bands
    Returns:
        the adapted conv layer
    """
    old_in_channels = old_conv.in_channels

    # Create new conv layer on the SAME DEVICE as the old one
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    ).to(old_conv.weight.device)  # Move to same device as original

    old_weights = old_conv.weight.data
    new_weights = torch.zeros(
        old_conv.out_channels,
        new_in_channels,
        *old_conv.kernel_size,
        device=old_weights.device,  # Use same device
        dtype=old_weights.dtype,
    )

    # Preserve existing band weights
    if new_in_channels >= old_in_channels:
        new_weights[:, :old_in_channels, :, :] = old_weights

        # For new bands, use average of existing band weights
        if new_in_channels > old_in_channels:
            train_bands = training_bands or ["B04", "B03", "B02"]
            for i in range(old_in_channels, new_in_channels):
                new_band = (
                    new_bands[i - old_in_channels] if new_bands and i - old_in_channels < len(new_bands) else None
                )
                if spectral_init and new_band:
                    w = get_spectral_init_weights(new_band, train_bands)
                    init_w = sum(old_weights[:, j, :, :] * wt for j, wt in w.items())
                else:
                    init_w = old_weights.mean(dim=1)
                new_weights[:, i, :, :] = init_w
    else:
        new_weights = old_weights[:, :new_in_channels, :, :].clone()
    new_conv.weight.data.copy_(new_weights)

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)

    return new_conv


def adapt_rgb_conv3d_layer_to_multiband(old_conv: nn.Conv3d, new_in_channels: int) -> nn.Conv3d:

    new_conv = nn.Conv3d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    old_weights = old_conv.weight.data
    averaged_weights = old_weights.mean(dim=1, keepdim=True)
    new_weights = averaged_weights.repeat(1, new_in_channels, 1, 1, 1)
    new_weights = new_weights / new_in_channels
    new_conv.weight.data.copy_(new_weights)

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)

    return new_conv


def adapt_rgb_conv3d_layer_to_multiband_preserve_rgb(
    old_conv: nn.Conv3d,
    new_in_channels: int,
    spectral_init: bool = False,
    training_bands: list[str] | None = None,
    new_bands: list[str] | None = None,
) -> nn.Conv3d:
    """
    Adapt a 3-channel RGB conv3d layer to a new number of channels, preserving the original weights.
    For each new channel, initialize its weights as a weighted average of the training bands.

    Args:
        old_conv: Original 3D convolution layer
        new_in_channels: New number of input channels
        spectral_init: whether to use spectral initialization
        training_bands: the training bands
        new_bands: the new bands
    Returns:
        the adapted 3D conv layer
    """
    old_in_channels = old_conv.in_channels

    new_conv = nn.Conv3d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    ).to(old_conv.weight.device)
    old_weights = old_conv.weight.data
    new_weights = torch.zeros(
        old_conv.out_channels,
        new_in_channels,
        *old_conv.kernel_size,
        device=old_weights.device,
        dtype=old_weights.dtype,
    )

    # Preserve existing band weights
    if new_in_channels >= old_in_channels:
        new_weights[:, :old_in_channels, :, :, :] = old_weights

        # For new bands, use average of existing band weights
        if new_in_channels > old_in_channels:
            train_bands = training_bands or ["B04", "B03", "B02"]
            for i in range(old_in_channels, new_in_channels):
                new_band = (
                    new_bands[i - old_in_channels] if new_bands and i - old_in_channels < len(new_bands) else None
                )
                if spectral_init and new_band:
                    w = get_spectral_init_weights(new_band, train_bands)
                    init_w = sum(old_weights[:, j, :, :, :] * wt for j, wt in w.items())
                else:
                    init_w = old_weights.mean(dim=1)
                new_weights[:, i, :, :, :] = init_w
    else:
        new_weights = old_weights[:, :new_in_channels, :, :, :].clone()
    new_conv.weight.data.copy_(new_weights)

    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)

    return new_conv


def adapt_encoder_for_multiband_eval(
    encoder,
    multiband_channel_count: int = 4,
    spectral_init: bool = False,
    training_bands: list[str] | None = None,
    new_bands: list[str] | None = None,
):
    """
    Adapt an encoder for multiband evaluation.

    Args:
        encoder: The encoder to adapt
        multiband_channel_count: The number of channels in the multiband input
        spectral_init: Whether to use spectral initialization
        training_bands: The training bands
        new_bands: The new bands
    Returns:
        True if the encoder was successfully adapted, False otherwise
    """
    adapt_kw = dict(
        new_in_channels=multiband_channel_count,
        spectral_init=spectral_init,
        training_bands=training_bands,
        new_bands=new_bands,
    )
    if hasattr(encoder, "patch_embed") and hasattr(encoder.patch_embed, "proj"):
        # Standard ViT, Swin, etc.
        old_conv = encoder.patch_embed.proj
        adapt_fn = (
            adapt_rgb_conv3d_layer_to_multiband_preserve_rgb
            if isinstance(old_conv, nn.Conv3d)
            else adapt_rgb_conv_layer_to_multiband_preserve_rgb
        )
        encoder.patch_embed.proj = adapt_fn(old_conv=old_conv, **adapt_kw)
        # Update input channel count
        if hasattr(encoder.patch_embed, "in_chans"):
            encoder.patch_embed.in_chans = multiband_channel_count
        if hasattr(encoder, "in_chans"):
            encoder.in_chans = multiband_channel_count

    elif (
        hasattr(encoder, "model")
        and hasattr(encoder.model, "patch_embed")
        and hasattr(encoder.model.patch_embed, "proj")
    ):
        # timm ViT, some other wrapped models
        old_conv = encoder.model.patch_embed.proj
        adapt_fn = (
            adapt_rgb_conv3d_layer_to_multiband_preserve_rgb
            if isinstance(old_conv, nn.Conv3d)
            else adapt_rgb_conv_layer_to_multiband_preserve_rgb
        )
        encoder.model.patch_embed.proj = adapt_fn(old_conv=old_conv, **adapt_kw)
        # Update input channel count
        if hasattr(encoder.model.patch_embed, "in_chans"):
            encoder.model.patch_embed.in_chans = multiband_channel_count
        if hasattr(encoder.model, "in_chans"):
            encoder.model.in_chans = multiband_channel_count

    elif (
        hasattr(encoder, "model")
        and hasattr(encoder.model, "model")
        and hasattr(encoder.model.model, "patch_embed")
        and hasattr(encoder.model.model.patch_embed, "proj")
    ):
        old_conv = encoder.model.model.patch_embed.proj
        adapt_fn = (
            adapt_rgb_conv3d_layer_to_multiband_preserve_rgb
            if isinstance(old_conv, nn.Conv3d)
            else adapt_rgb_conv_layer_to_multiband_preserve_rgb
        )
        encoder.model.model.patch_embed.proj = adapt_fn(old_conv=old_conv, **adapt_kw)
        if hasattr(encoder.model.model.patch_embed, "in_chans"):
            encoder.model.model.patch_embed.in_chans = multiband_channel_count

    elif (
        hasattr(encoder, "backbone")
        and hasattr(encoder.backbone, "patch_embed")
        and hasattr(encoder.backbone.patch_embed, "proj")
    ):
        old_conv = encoder.backbone.patch_embed.proj
        adapt_fn = (
            adapt_rgb_conv3d_layer_to_multiband_preserve_rgb
            if isinstance(old_conv, nn.Conv3d)
            else adapt_rgb_conv_layer_to_multiband_preserve_rgb
        )
        encoder.backbone.patch_embed.proj = adapt_fn(old_conv=old_conv, **adapt_kw)
        # Update input channel count
        if hasattr(encoder.backbone.patch_embed, "in_chans"):
            encoder.backbone.patch_embed.in_chans = multiband_channel_count
        if hasattr(encoder.backbone, "in_chans"):
            encoder.backbone.in_chans = multiband_channel_count

    elif (
        hasattr(encoder, "backbone")
        and hasattr(encoder.backbone, "backbone")
        and hasattr(encoder.backbone.backbone, "patch_embed")
        and hasattr(encoder.backbone.backbone.patch_embed, "proj")
    ):
        old_conv = encoder.backbone.backbone.patch_embed.proj
        adapt_fn = (
            adapt_rgb_conv3d_layer_to_multiband_preserve_rgb
            if isinstance(old_conv, nn.Conv3d)
            else adapt_rgb_conv_layer_to_multiband_preserve_rgb
        )
        encoder.backbone.backbone.patch_embed.proj = adapt_fn(old_conv=old_conv, **adapt_kw)
        # Update input channel count
        if hasattr(encoder.backbone.backbone.patch_embed, "in_chans"):
            encoder.backbone.backbone.patch_embed.in_chans = multiband_channel_count

    elif (
        hasattr(encoder, "backbone")
        and hasattr(encoder.backbone, "features")
        and hasattr(encoder.backbone.features, "[0]")
        and hasattr(encoder.backbone.features[0], "[0]")
    ):
        # Swin transformer with features
        old_conv = encoder.backbone.features[0][0]
        encoder.backbone.features[0][0] = adapt_rgb_conv_layer_to_multiband_preserve_rgb(old_conv=old_conv, **adapt_kw)
        # Update input channel count
        if hasattr(encoder.backbone, "in_channels"):
            encoder.backbone.in_channels = multiband_channel_count

    elif hasattr(encoder, "model") and hasattr(encoder.model, "conv1"):
        # ResNet-style models
        old_conv = encoder.model.conv1
        encoder.model.conv1 = adapt_rgb_conv_layer_to_multiband_preserve_rgb(old_conv=old_conv, **adapt_kw)
        # Update input channel count
        if hasattr(encoder.model, "in_channels"):
            encoder.model.in_channels = multiband_channel_count

    elif hasattr(encoder, "embeddings") and hasattr(encoder.embeddings, "patch_embeddings"):
        # DINOv3 from transformers
        old_conv = encoder.embeddings.patch_embeddings
        encoder.embeddings.patch_embeddings = adapt_rgb_conv_layer_to_multiband_preserve_rgb(
            old_conv=old_conv, **adapt_kw
        )
        # Update input channel count in config if available
        if hasattr(encoder.config, "num_channels"):
            encoder.config.num_channels = multiband_channel_count

    else:
        print(f"Warning: Could not find conv layer to adapt in encoder type: {type(encoder)}")
        print(f"Available attributes: {[attr for attr in dir(encoder) if not attr.startswith('_')]}")
        return False

    print(f"Successfully adapted encoder to {multiband_channel_count} channels")


def load_encoder(
    encoder_name="ibot-B",
    encoder_weights="imagenet",
    enable_sample=False,
    shared_proj=False,
    add_ch_embed=False,
    color_blind=False,
    enable_multiband_input=False,
    multiband_channel_count=12,
    min_sample_channels: int = 1,
    pooling_mode: str = "cls",
    enable_channel_gate: bool = False,
):

    if "timm" in encoder_name.lower():
        Encoder = timm_encoders[encoder_name]["encoder"]
        params = timm_encoders[encoder_name]["params"]
        params.update(for_cls=True)

        if enable_multiband_input:
            params["in_channels"] = multiband_channel_count

        encoder = Encoder(**params)

        if enable_multiband_input and hasattr(encoder.model, "conv1"):
            old_conv = encoder.model.conv1
            encoder.model.conv1 = adapt_rgb_conv_layer_to_multiband(
                old_conv=old_conv, new_in_channels=multiband_channel_count
            )
        elif (
            enable_multiband_input
            and hasattr(encoder.model, "patch_embed")
            and hasattr(encoder.model.patch_embed, "proj")
        ):
            old_conv = encoder.model.patch_embed.proj
            encoder.model.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                old_conv=old_conv, new_in_channels=multiband_channel_count
            )

        return encoder

    elif "swin" in encoder_name.lower():
        if "satlas_ms" in encoder_weights.lower():
            import satlaspretrain_models

            weights_manager = satlaspretrain_models.Weights()
            encoder = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_SI_MS")

            if enable_multiband_input:
                old_conv = encoder.backbone.backbone.features[0][0]
                encoder.backbone.backbone.features[0][0] = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, new_in_channels=multiband_channel_count
                )
        else:
            Encoder = swin_transformer_encoders[encoder_name]["encoder"]
            params = swin_transformer_encoders[encoder_name]["params"]
            gap = False if "satlas" in encoder_weights else True
            params.update(for_cls=True, gap=gap, window_size=8)

            if enable_multiband_input:
                params["in_channels"] = multiband_channel_count

            encoder = Encoder(**params)
            settings = swin_transformer_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            checkmoint_model = load_pretrained(encoder, settings["url"], "cpu")
            msg = encoder.load_state_dict(checkmoint_model, strict=False)
            print(msg)

            if enable_multiband_input:
                old_conv = encoder.patch_embed.proj
                encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, new_in_channels=multiband_channel_count
                )

    elif "ibot" in encoder_name.lower():
        Encoder = vit_encoders[encoder_name]["encoder"]
        params = vit_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)
        if encoder_weights == "random":
            return encoder
        else:
            settings = vit_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            if "imagenet" in settings["url"]:
                state_dict = torch.load(settings["url"], map_location=torch.device("cpu"))["state_dict"]
            else:
                state_dict = torch.load(settings["url"], map_location=torch.device("cpu"))["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = encoder.load_state_dict(state_dict, strict=False)
            print(msg)
            if enable_multiband_input:
                old_conv = encoder.patch_embed.proj
                encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, new_in_channels=multiband_channel_count
                )
    elif "dino" in encoder_name.lower():
        if "sat" in encoder_name.lower():
            Encoder = dinov2_encoders[encoder_name]["encoder"]
            params = dinov2_encoders[encoder_name]["params"]
            params.update(classification=True)
            params.update(color_blind=color_blind)
            encoder = Encoder(**params).eval()

            if enable_multiband_input:
                if (
                    hasattr(encoder, "backbone")
                    and hasattr(encoder.backbone, "backbone")
                    and hasattr(encoder.backbone.backbone, "patch_embed")
                    and hasattr(encoder.backbone.backbone.patch_embed, "proj")
                ):
                    old_conv = encoder.backbone.backbone.patch_embed.proj
                    encoder.backbone.backbone.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                        old_conv=old_conv, new_in_channels=multiband_channel_count
                    )
            # path = '/nfs/ap/mnt/frtn/rs-results/dinov2_sat/SSLhuge_satellite.pth'
            # encoder = SSLAE(pretrained=path, huge=True, classification=True).eval()
        elif "v3" in encoder_name.lower():
            print("=" * 100)
            print("Loading Dinov3 encoder")
            print("=" * 100)
            encoder = timm.create_model(
                "vit_base_patch16_dinov3.lvd1689m",
                pretrained=True,
                num_classes=0,
                global_pool="avg",
                dynamic_img_size=True,
            ).eval()

            if color_blind:
                old_conv = encoder.patch_embed.proj
                k = (
                    old_conv.kernel_size
                    if isinstance(old_conv.kernel_size, tuple)
                    else (old_conv.kernel_size, old_conv.kernel_size)
                )
                s = old_conv.stride if isinstance(old_conv.stride, tuple) else (old_conv.stride, old_conv.stride)
                p = old_conv.padding if isinstance(old_conv.padding, tuple) else (old_conv.padding, old_conv.padding)
                mod = SharedChannelPatchConv(
                    out_channels=old_conv.out_channels,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=(old_conv.bias is not None),
                    in_channels=old_conv.in_channels,
                    device=old_conv.weight.device,
                )
                w_avg = old_conv.weight.data.mean(dim=1, keepdim=True)
                mod.weight.data.copy_(w_avg)
                if old_conv.bias is not None:
                    mod.bias.data.copy_(old_conv.bias.data)
                encoder.patch_embed.proj = mod

            if enable_multiband_input:
                pe = encoder.patch_embed.proj
                if isinstance(pe, nn.Conv2d):
                    encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                        old_conv=pe, new_in_channels=multiband_channel_count
                    )
                elif isinstance(pe, SharedChannelPatchConv):
                    pe.in_channels = multiband_channel_count
        else:
            try:
                encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
            except TypeError:
                encoder = timm.create_model(
                    "vit_base_patch14_dinov2.lvd142m",
                    pretrained=True,
                    num_classes=0,
                    dynamic_img_size=True,
                ).eval()

            if enable_multiband_input:
                old_conv = encoder.patch_embed.proj
                encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                    old_conv=old_conv, new_in_channels=multiband_channel_count
                )

    elif "cvit-pretrained" in encoder_name.lower():
        Encoder = chi_vit_encoders[encoder_name]["encoder"]
        params = chi_vit_encoders[encoder_name]["params"]
        params.update(return_feats=False)
        params.update(enable_sample=enable_sample)
        params.update(shared_proj=shared_proj)
        params.update(add_ch_embed=add_ch_embed)
        params.update(min_sample_channels=min_sample_channels)
        params.update(pooling_mode=pooling_mode)
        params.update(enable_channel_gate=enable_channel_gate)

        encoder = Encoder(**params)

        # Load weights
        settings = chi_vit_encoders[encoder_name]["pretrained_settings"][encoder_weights]
        state_dict = torch.load(settings["url"], map_location=torch.device("cpu"), weights_only=False)["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(msg)

    elif "cvit" in encoder_name.lower():
        encoder = torch.hub.load(
            "insitro/ChannelViT", "so2sat_channelvit_small_p8_with_hcs_random_split_supervised", pretrained=True
        )

    elif "anysat" in encoder_name.lower():
        Encoder = anysat_encoders[encoder_name]["encoder"]
        params = anysat_encoders[encoder_name]["params"].copy()
        params["for_cls"] = True
        params["out_idx"] = None
        params["out_channels"] = None
        encoder = Encoder(**params)
        pretrained_encoder = encoder.from_pretrained("base", flash_attn=False)
        encoder.model.load_state_dict(pretrained_encoder.model.state_dict(), strict=False)

    elif "croma" in encoder_name.lower():
        Encoder = croma_encoders[encoder_name]["encoder"]
        params = croma_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)

    elif "terrafm" in encoder_name.lower():
        Encoder = terrafm_encoders[encoder_name]["encoder"]
        params = terrafm_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)

        settings = terrafm_encoders[encoder_name]["pretrained_settings"][encoder_weights]
        state_dict = torch.load(settings["url"], map_location=torch.device("cpu"))
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded TerraFM pretrained weights from {settings['url']}")
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")

        if enable_multiband_input:
            if hasattr(encoder.patch_embed, "proj"):
                old_conv = encoder.patch_embed.proj
                if old_conv.in_channels != multiband_channel_count:
                    encoder.patch_embed.proj = adapt_rgb_conv_layer_to_multiband(
                        old_conv=old_conv,
                        new_in_channels=multiband_channel_count,
                    )
            elif hasattr(encoder.patch_embed, "conv2d_s2_l1c") and hasattr(encoder.patch_embed, "conv2d_s2_l2a"):
                old_conv_l1c = encoder.patch_embed.conv2d_s2_l1c
                old_conv_l2a = encoder.patch_embed.conv2d_s2_l2a
                if old_conv_l1c.in_channels != multiband_channel_count:
                    encoder.patch_embed.conv2d_s2_l1c = adapt_rgb_conv_layer_to_multiband(
                        old_conv=old_conv_l1c,
                        new_in_channels=multiband_channel_count,
                    )
                if old_conv_l2a.in_channels != multiband_channel_count:
                    encoder.patch_embed.conv2d_s2_l2a = adapt_rgb_conv_layer_to_multiband(
                        old_conv=old_conv_l2a,
                        new_in_channels=multiband_channel_count,
                    )
            else:
                raise ValueError(f"Unsupported TerraFM patch embed for multiband: {type(encoder.patch_embed)}")

    elif "prithvi" in encoder_name.lower():
        Encoder = prithvi_encoders[encoder_name]["encoder"]
        params = prithvi_encoders[encoder_name]["params"]
        params.update(for_cls=True)

        encoder = Encoder(**params)
        settings = prithvi_encoders[encoder_name]["pretrained_settings"][encoder_weights]
        state_dict = torch.load(settings["url"], map_location=torch.device("cpu"))
        del state_dict["pos_embed"]
        del state_dict["decoder_pos_embed"]

        msg = encoder.load_state_dict(state_dict, strict=False)
        print(msg)

        if enable_multiband_input:
            old_conv = encoder.patch_embed.proj
            encoder.patch_embed.proj = adapt_rgb_conv3d_layer_to_multiband(
                old_conv=old_conv, new_in_channels=multiband_channel_count
            )
            params["in_chans"] = multiband_channel_count

    elif "clay" in encoder_name.lower():
        Encoder = clay_encoders[encoder_name]["encoder"]
        params = clay_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        encoder = Encoder(**params)

    elif "dofa" in encoder_name.lower():
        Encoder = dofa_encoders[encoder_name]["encoder"]
        params = dofa_encoders[encoder_name]["params"]
        params.update(for_cls=True)
        params.update(global_pool=False)
        encoder = Encoder(**params)

        settings = dofa_encoders[encoder_name]["pretrained_settings"][encoder_weights]
        state_dict = torch.load(settings["url"], map_location=torch.device("cpu"))
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(msg)

    return encoder
