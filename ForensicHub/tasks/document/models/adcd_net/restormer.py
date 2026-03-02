"""
Restormer architecture for ADCD-Net
Based on "Restormer: Efficient Transformer for High-Resolution Image Restoration"
https://arxiv.org/abs/2111.09881

Modified for document tampering detection with DCT feature integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers


def to_3d(x):
    """Convert (B, C, H, W) to (B, H*W, C)"""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x, h, w):
    """Convert (B, H*W, C) to (B, C, H, W)"""
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        del x
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape: (B, C, H, W) -> (B, num_heads, C//num_heads, H*W)
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        # Reshape back: (B, num_heads, C//num_heads, H*W) -> (B, C, H, W)
        out = out.reshape(b, c, h, w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class DCTFusion(nn.Module):
    """Fuse DCT features with RGB features using alignment score."""

    def __init__(self, rgb_dim, dct_dim):
        super().__init__()
        self.rgb_proj = nn.Conv2d(rgb_dim, rgb_dim, 1)
        self.dct_proj = nn.Conv2d(dct_dim, rgb_dim, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(rgb_dim * 2, rgb_dim, 3, 1, 1),
            nn.BatchNorm2d(rgb_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, dct_feat, align_score):
        """
        Args:
            rgb_feat: RGB features (B, C_rgb, H, W)
            dct_feat: DCT features (B, C_dct, H', W')
            align_score: Alignment score (B,) with values 0-1
        """
        # Resize DCT features to match RGB features
        if dct_feat.shape[-2:] != rgb_feat.shape[-2:]:
            dct_feat = F.interpolate(dct_feat, size=rgb_feat.shape[-2:], mode='bilinear', align_corners=False)

        rgb_proj = self.rgb_proj(rgb_feat)
        dct_proj = self.dct_proj(dct_feat)

        # Modulate DCT contribution by alignment score
        align_score = align_score.view(-1, 1, 1, 1)
        dct_modulated = dct_proj * align_score

        fused = self.fusion(torch.cat([rgb_proj, dct_modulated], dim=1))
        return fused + rgb_feat


class Restormer(nn.Module):
    """
    Restormer-based encoder-decoder for ADCD-Net.

    Modified to output multi-scale features for:
    - Localization branch
    - Content/foreground disentanglement
    - Pristine prototype estimation
    """

    def __init__(
            self,
            inp_channels=6,  # RGB (3) + coord (2) + placeholder (1)
            out_channels=96,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dct_dims=[128, 64, 48, 48]  # DCT feature dimensions at each scale
    ):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[3])
        ])

        # DCT fusion modules
        self.dct_fusion1 = DCTFusion(dim, dct_dims[0])
        self.dct_fusion2 = DCTFusion(int(dim * 2), dct_dims[1])
        self.dct_fusion3 = DCTFusion(int(dim * 4), dct_dims[2])
        self.dct_fusion4 = DCTFusion(int(dim * 8), dct_dims[3])

        # Decoder
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # Feature extractors for disentanglement
        self.cnt_head = nn.Conv2d(int(dim * 2 ** 1), out_channels, 3, 1, 1)
        self.frg_head = nn.Conv2d(int(dim * 2 ** 1), out_channels, 3, 1, 1)

    def forward(self, img, ms_dct_feats=None, dct_align_score=None):
        """
        Args:
            img: Input image with coordinates (B, 6, H, W)
            ms_dct_feats: Multi-scale DCT features from FPH
            dct_align_score: Alignment score from FPH (B,)

        Returns:
            feat: Main localization feature (B, out_channels, H, W)
            cnt_feats: Content features for reconstruction
            frg_feats: Foreground features for reconstruction
            pp_feats: Pristine prototype features at multiple scales
        """
        inp_enc_level1 = self.patch_embed(img)

        # Encoder with DCT fusion
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        if ms_dct_feats is not None and dct_align_score is not None:
            out_enc_level1 = self.dct_fusion1(out_enc_level1, ms_dct_feats[0], dct_align_score)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        if ms_dct_feats is not None and dct_align_score is not None:
            out_enc_level2 = self.dct_fusion2(out_enc_level2, ms_dct_feats[1], dct_align_score)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        if ms_dct_feats is not None and dct_align_score is not None:
            out_enc_level3 = self.dct_fusion3(out_enc_level3, ms_dct_feats[2], dct_align_score)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        if ms_dct_feats is not None and dct_align_score is not None:
            latent = self.dct_fusion4(latent, ms_dct_feats[3], dct_align_score)

        # Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        feat = self.output(out_dec_level1)

        # Content and foreground features for reconstruction
        cnt_feats = self.cnt_head(out_dec_level1)
        frg_feats = self.frg_head(out_dec_level1)

        # Pristine prototype features at multiple scales
        pp_feats = [
            F.interpolate(out_dec_level1, scale_factor=0.5, mode='bilinear', align_corners=False),
            out_dec_level2,
            out_dec_level3,
            latent
        ]

        return feat, cnt_feats, frg_feats, pp_feats


class RestormerDecoder(nn.Module):
    """Decoder-only Restormer for reconstruction branch."""

    def __init__(self, in_channels=96, out_channels=4, dim=48):
        super().__init__()

        self.proj = nn.Conv2d(in_channels * 2, dim * 2, 1)

        self.decoder = nn.Sequential(
            TransformerBlock(dim * 2, num_heads=2),
            TransformerBlock(dim * 2, num_heads=2),
            Upsample(dim * 2),
            TransformerBlock(dim, num_heads=1),
            TransformerBlock(dim, num_heads=1),
        )

        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, cnt_feats, frg_feats, is_shuffle=False):
        """
        Reconstruct image from content and foreground features.

        Args:
            cnt_feats: Content features (B, C, H, W)
            frg_feats: Foreground features (B, C, H, W)
            is_shuffle: Whether to shuffle foreground features for contrastive learning
        """
        if is_shuffle:
            # Shuffle foreground features across batch
            idx = torch.randperm(frg_feats.size(0))
            frg_feats = frg_feats[idx]

        x = torch.cat([cnt_feats, frg_feats], dim=1)
        x = self.proj(x)
        x = self.decoder(x)
        return self.output(x)


def get_restormer(model_name='full_model', out_channels=96, **kwargs):
    """
    Factory function to create Restormer models.

    Args:
        model_name: 'full_model' for encoder-decoder, 'decoder_only' for reconstruction
        out_channels: Number of output channels
    """
    if model_name == 'full_model':
        return Restormer(out_channels=out_channels, **kwargs)
    elif model_name == 'decoder_only':
        return RestormerDecoder(out_channels=out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # Test full model
    model = get_restormer('full_model', out_channels=96)
    img = torch.randn(2, 6, 256, 256)
    ms_dct_feats = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 48, 8, 8),
        torch.randn(2, 48, 4, 4),
    ]
    align_score = torch.rand(2)

    feat, cnt_feats, frg_feats, pp_feats = model(img, ms_dct_feats, align_score)
    print(f"Main feature: {feat.shape}")
    print(f"Content features: {cnt_feats.shape}")
    print(f"Foreground features: {frg_feats.shape}")
    for i, f in enumerate(pp_feats):
        print(f"PP feature {i}: {f.shape}")

    # Test decoder
    decoder = get_restormer('decoder_only', out_channels=4)
    rec = decoder(cnt_feats, frg_feats, is_shuffle=True)
    print(f"Reconstructed: {rec.shape}")